"""
Debug utilities for VLM (Vision-Language Model) integration debugging.

This module provides tools to visualize and debug what's being sent to VLMs,
helping diagnose issues like:
- High loss that doesn't decrease
- Extreme gradient norms
- Numerical instability
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TensorStats:
    """Statistics for a tensor."""
    name: str
    shape: tuple
    dtype: str
    device: str
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    has_nan: bool
    has_inf: bool
    num_zeros: int

    def __str__(self) -> str:
        status = "OK"
        if self.has_nan:
            status = "⚠️ HAS NaN"
        elif self.has_inf:
            status = "⚠️ HAS Inf"
        elif self.std_val == 0:
            status = "⚠️ ZERO STD"

        return (
            f"{self.name}: shape={self.shape}, dtype={self.dtype}, "
            f"range=[{self.min_val:.4g}, {self.max_val:.4g}], "
            f"mean={self.mean_val:.4g}, std={self.std_val:.4g}, "
            f"zeros={self.num_zeros}, {status}"
        )


def compute_tensor_stats(tensor: torch.Tensor, name: str = "tensor") -> TensorStats:
    """Compute statistics for a tensor."""
    with torch.no_grad():
        t = tensor.float()
        return TensorStats(
            name=name,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            min_val=t.min().item(),
            max_val=t.max().item(),
            mean_val=t.mean().item(),
            std_val=t.std().item() if t.numel() > 1 else 0.0,
            has_nan=torch.isnan(t).any().item(),
            has_inf=torch.isinf(t).any().item(),
            num_zeros=(t == 0).sum().item(),
        )


def log_tensor_stats(tensor: torch.Tensor, name: str = "tensor", level: int = logging.DEBUG) -> TensorStats:
    """Log statistics for a tensor and return the stats."""
    stats = compute_tensor_stats(tensor, name)
    logging.log(level, str(stats))
    return stats


@dataclass
class GradientStats:
    """Statistics for gradients of a parameter group."""
    group_name: str
    num_params: int
    total_elements: int
    grad_norm: float
    max_grad: float
    min_grad: float
    mean_grad: float
    has_nan: bool
    has_inf: bool
    params_with_grad: int

    def __str__(self) -> str:
        status = "OK"
        if self.has_nan:
            status = "⚠️ HAS NaN"
        elif self.has_inf:
            status = "⚠️ HAS Inf"
        elif self.grad_norm > 1e6:
            status = "⚠️ EXTREME NORM"

        return (
            f"{self.group_name}: params={self.num_params}, elements={self.total_elements}, "
            f"grad_norm={self.grad_norm:.4g}, range=[{self.min_grad:.4g}, {self.max_grad:.4g}], "
            f"mean={self.mean_grad:.4g}, with_grad={self.params_with_grad}, {status}"
        )


def compute_gradient_stats_for_module(
    module: nn.Module,
    group_name: str = "module"
) -> GradientStats:
    """Compute gradient statistics for all parameters in a module."""
    grads = []
    num_params = 0
    params_with_grad = 0

    for param in module.parameters():
        num_params += 1
        if param.grad is not None:
            params_with_grad += 1
            grads.append(param.grad.detach().float().flatten())

    if not grads:
        return GradientStats(
            group_name=group_name,
            num_params=num_params,
            total_elements=0,
            grad_norm=0.0,
            max_grad=0.0,
            min_grad=0.0,
            mean_grad=0.0,
            has_nan=False,
            has_inf=False,
            params_with_grad=0,
        )

    all_grads = torch.cat(grads)
    grad_norm = all_grads.norm().item()

    return GradientStats(
        group_name=group_name,
        num_params=num_params,
        total_elements=all_grads.numel(),
        grad_norm=grad_norm,
        max_grad=all_grads.max().item(),
        min_grad=all_grads.min().item(),
        mean_grad=all_grads.mean().item(),
        has_nan=torch.isnan(all_grads).any().item(),
        has_inf=torch.isinf(all_grads).any().item(),
        params_with_grad=params_with_grad,
    )


def compute_gradient_stats_by_group(
    model: nn.Module,
    group_patterns: Optional[dict[str, str]] = None,
) -> dict[str, GradientStats]:
    """
    Compute gradient statistics grouped by parameter name patterns.

    Args:
        model: The model to analyze
        group_patterns: Dict mapping group names to regex patterns for parameter names.
                       If None, uses default VLM patterns.

    Returns:
        Dict mapping group names to GradientStats
    """
    import re

    if group_patterns is None:
        # Default patterns for SmolVLA2/Gemma integration
        # These patterns match the actual parameter names in the model
        group_patterns = {
            "vision": r"(vision_model|vision_tower|vision)",
            "connector": r"(connector|multi_modal_projector)",
            "vlm_text": r"(text_model|language_model|vlm.*model\.layers)",
            "lm_expert": r"lm_expert",
            "state_proj": r"state_proj",
            "action_proj": r"action_(in|out)_proj",
            "action_time_mlp": r"action_time_mlp",
            "lora": r"lora_",
            "other": r".*",  # Catch-all
        }

    # Compile patterns
    compiled_patterns = {name: re.compile(pattern) for name, pattern in group_patterns.items()}

    # Group parameters
    grouped_grads: dict[str, list[torch.Tensor]] = {name: [] for name in group_patterns}
    grouped_counts: dict[str, int] = {name: 0 for name in group_patterns}
    matched_params: set[str] = set()

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        for group_name, pattern in compiled_patterns.items():
            if group_name == "other":
                continue
            if pattern.search(name):
                grouped_grads[group_name].append(param.grad.detach().float().flatten())
                grouped_counts[group_name] += 1
                matched_params.add(name)
                break
        else:
            # No pattern matched, add to "other"
            if "other" in grouped_grads:
                grouped_grads["other"].append(param.grad.detach().float().flatten())
                grouped_counts["other"] += 1

    # Compute stats for each group
    results = {}
    for group_name, grads in grouped_grads.items():
        if not grads:
            results[group_name] = GradientStats(
                group_name=group_name,
                num_params=grouped_counts[group_name],
                total_elements=0,
                grad_norm=0.0,
                max_grad=0.0,
                min_grad=0.0,
                mean_grad=0.0,
                has_nan=False,
                has_inf=False,
                params_with_grad=0,
            )
            continue

        all_grads = torch.cat(grads)
        results[group_name] = GradientStats(
            group_name=group_name,
            num_params=grouped_counts[group_name],
            total_elements=all_grads.numel(),
            grad_norm=all_grads.norm().item(),
            max_grad=all_grads.max().item(),
            min_grad=all_grads.min().item(),
            mean_grad=all_grads.mean().item(),
            has_nan=torch.isnan(all_grads).any().item(),
            has_inf=torch.isinf(all_grads).any().item(),
            params_with_grad=len(grads),
        )

    return results


def log_gradient_stats_by_group(
    model: nn.Module,
    step: int,
    group_patterns: Optional[dict[str, str]] = None,
    level: int = logging.INFO,
) -> dict[str, GradientStats]:
    """Log gradient statistics grouped by parameter patterns."""
    stats = compute_gradient_stats_by_group(model, group_patterns)

    # Compute total stats
    total_grad_norm = 0.0
    total_with_grad = 0
    has_any_nan = False
    has_any_inf = False

    for group_name, group_stats in stats.items():
        if group_stats.params_with_grad > 0:
            total_grad_norm += group_stats.grad_norm ** 2
            total_with_grad += group_stats.params_with_grad
            has_any_nan = has_any_nan or group_stats.has_nan
            has_any_inf = has_any_inf or group_stats.has_inf

    total_grad_norm = total_grad_norm ** 0.5

    logging.log(level, f"=== Gradient Stats Step {step}: total_norm={total_grad_norm:.4g}, params={total_with_grad} ===")
    for group_name, group_stats in stats.items():
        if group_stats.params_with_grad > 0:
            logging.log(level, f"  {group_stats}")

    if has_any_nan:
        logging.error(f"Step {step}: NaN detected in gradients!")
    if has_any_inf:
        logging.error(f"Step {step}: Inf detected in gradients!")

    return stats


class VLMDebugger:
    """
    Debug helper for VLM integration issues.

    Usage:
        debugger = VLMDebugger(output_dir="debug_outputs", enabled=True)

        # In forward pass:
        debugger.log_embeddings(
            step=step,
            image_embs=image_embs,
            lang_embs=lang_embs,
            state_embs=state_embs,
        )

        # After backward:
        debugger.log_gradients(step=step, model=model)
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        enabled: bool = False,
        log_every_n_steps: int = 100,
        save_tensors: bool = False,
    ):
        self.enabled = enabled
        self.output_dir = Path(output_dir) if output_dir else None
        self.log_every_n_steps = log_every_n_steps
        self.save_tensors = save_tensors

        if self.enabled and self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def should_log(self, step: int) -> bool:
        """Check if we should log at this step."""
        return self.enabled and (step % self.log_every_n_steps == 0)

    def log_embeddings(
        self,
        step: int,
        image_embs: Optional[list[torch.Tensor]] = None,
        lang_embs: Optional[torch.Tensor] = None,
        state_embs: Optional[torch.Tensor] = None,
        action_embs: Optional[torch.Tensor] = None,
        prefix_embs: Optional[torch.Tensor] = None,
        suffix_embs: Optional[torch.Tensor] = None,
        pad_masks: Optional[torch.Tensor] = None,
        att_masks: Optional[torch.Tensor] = None,
    ) -> dict[str, TensorStats]:
        """Log embedding statistics."""
        if not self.should_log(step):
            return {}

        stats = {}
        logging.info(f"=== Embedding Stats at Step {step} ===")

        if image_embs is not None:
            for i, img_emb in enumerate(image_embs):
                s = log_tensor_stats(img_emb, f"image_emb_{i}", logging.INFO)
                stats[f"image_emb_{i}"] = s

        if lang_embs is not None:
            stats["lang_embs"] = log_tensor_stats(lang_embs, "lang_embs", logging.INFO)

        if state_embs is not None:
            stats["state_embs"] = log_tensor_stats(state_embs, "state_embs", logging.INFO)

        if action_embs is not None:
            stats["action_embs"] = log_tensor_stats(action_embs, "action_embs", logging.INFO)

        if prefix_embs is not None:
            stats["prefix_embs"] = log_tensor_stats(prefix_embs, "prefix_embs", logging.INFO)

        if suffix_embs is not None:
            stats["suffix_embs"] = log_tensor_stats(suffix_embs, "suffix_embs", logging.INFO)

        if pad_masks is not None:
            stats["pad_masks"] = log_tensor_stats(pad_masks.float(), "pad_masks", logging.INFO)

        if att_masks is not None:
            stats["att_masks"] = log_tensor_stats(att_masks.float(), "att_masks", logging.INFO)

        # Save tensors if requested
        if self.save_tensors and self.output_dir:
            save_path = self.output_dir / f"embeddings_step_{step}.pt"
            torch.save({
                "image_embs": [e.detach().cpu() for e in image_embs] if image_embs else None,
                "lang_embs": lang_embs.detach().cpu() if lang_embs is not None else None,
                "state_embs": state_embs.detach().cpu() if state_embs is not None else None,
                "action_embs": action_embs.detach().cpu() if action_embs is not None else None,
                "prefix_embs": prefix_embs.detach().cpu() if prefix_embs is not None else None,
                "suffix_embs": suffix_embs.detach().cpu() if suffix_embs is not None else None,
                "pad_masks": pad_masks.detach().cpu() if pad_masks is not None else None,
                "att_masks": att_masks.detach().cpu() if att_masks is not None else None,
            }, save_path)
            logging.info(f"Saved embeddings to {save_path}")

        return stats

    def log_gradients(
        self,
        step: int,
        model: nn.Module,
        group_patterns: Optional[dict[str, str]] = None,
    ) -> dict[str, GradientStats]:
        """Log gradient statistics by parameter group."""
        if not self.should_log(step):
            return {}

        return log_gradient_stats_by_group(model, step, group_patterns, logging.INFO)

    def log_attention_weights(
        self,
        step: int,
        attention_weights: torch.Tensor,
        layer_idx: int,
        name: str = "attention",
    ) -> Optional[TensorStats]:
        """Log attention weight statistics."""
        if not self.should_log(step):
            return None

        stats = log_tensor_stats(
            attention_weights,
            f"{name}_layer{layer_idx}",
            logging.INFO
        )

        # Check for attention collapse (all weights going to one position)
        with torch.no_grad():
            max_attn = attention_weights.max(dim=-1).values
            if (max_attn > 0.99).float().mean() > 0.5:
                logging.warning(
                    f"⚠️ Attention collapse detected at layer {layer_idx}: "
                    f"{(max_attn > 0.99).float().mean():.1%} of positions have >99% attention to single token"
                )

        return stats

    def check_numerical_health(
        self,
        step: int,
        loss: torch.Tensor,
        grad_norm: float,
    ) -> dict[str, bool]:
        """Check for numerical issues and log warnings."""
        issues = {
            "loss_nan": False,
            "loss_inf": False,
            "loss_extreme": False,
            "grad_nan": False,
            "grad_extreme": False,
        }

        with torch.no_grad():
            if torch.isnan(loss):
                logging.error(f"Step {step}: Loss is NaN!")
                issues["loss_nan"] = True

            if torch.isinf(loss):
                logging.error(f"Step {step}: Loss is Inf!")
                issues["loss_inf"] = True

            if loss.item() > 100:
                logging.warning(f"Step {step}: Loss is very high: {loss.item():.4g}")
                issues["loss_extreme"] = True

        if grad_norm != grad_norm:  # NaN check
            logging.error(f"Step {step}: Gradient norm is NaN!")
            issues["grad_nan"] = True

        if grad_norm > 1e6:
            logging.warning(f"Step {step}: Gradient norm is extreme: {grad_norm:.4g}")
            issues["grad_extreme"] = True

        return issues


# Global debugger instance (can be configured at startup)
_global_debugger: Optional[VLMDebugger] = None


def get_vlm_debugger() -> Optional[VLMDebugger]:
    """Get the global VLM debugger instance."""
    return _global_debugger


def init_vlm_debugger(
    output_dir: Optional[str] = None,
    enabled: bool = False,
    log_every_n_steps: int = 100,
    save_tensors: bool = False,
) -> VLMDebugger:
    """Initialize the global VLM debugger."""
    global _global_debugger
    _global_debugger = VLMDebugger(
        output_dir=output_dir,
        enabled=enabled,
        log_every_n_steps=log_every_n_steps,
        save_tensors=save_tensors,
    )
    return _global_debugger
