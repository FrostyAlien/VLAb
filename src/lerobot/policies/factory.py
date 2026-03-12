#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
# from lerobot.envs.configs import EnvConfig  # Removed - not needed for SmolVLA pretraining
# from lerobot.envs.utils import env_to_policy_features  # Removed - not needed for SmolVLA pretraining
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla2.configuration_smolvla2 import SmolVLA2Config


def get_policy_class(name: str) -> PreTrainedPolicy:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""
    if name == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy
    elif name == "smolvla2":
        from lerobot.policies.smolvla2.modeling_smolvla2 import SmolVLA2Policy

        return SmolVLA2Policy
    else:
        raise NotImplementedError(
            f"Policy with name {name} is not implemented. Supported policies: smolvla, smolvla2."
        )


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    if policy_type == "smolvla":
        return SmolVLAConfig(**kwargs)
    elif policy_type == "smolvla2":
        return SmolVLA2Config(**kwargs)
    else:
        raise ValueError(f"Policy type '{policy_type}' is not available. Supported: smolvla, smolvla2.")


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
) -> PreTrainedPolicy:
    """Make an instance of a policy class.

    This function exists because (for now) we need to parse features from a dataset
    in order to properly dimension and instantiate a policy for that dataset.

    Args:
        cfg (PreTrainedConfig): The config of the policy to make. If `pretrained_path` is set, the policy will
            be loaded with the weights from that path.
        ds_meta (LeRobotDatasetMetadata | None, optional): Dataset metadata to take input/output shapes and
            statistics to use for (un)normalization of inputs/outputs in the policy. Defaults to None.

    Raises:
        ValueError: ds_meta must be provided.

    Returns:
        PreTrainedPolicy: SmolVLA policy instance
    """
    if not ds_meta:
        raise ValueError("Dataset metadata must be provided for SmolVLA pretraining.")

    # SmolVLA policy factory (smolvla / smolvla2) - no backend compatibility checks needed.

    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    # Parse features from dataset metadata (required for SmolVLA).
    features = dataset_to_policy_features(ds_meta.features)
    # Handle robot-type grouped stats - flatten to feature-level stats
    if ds_meta.stats and len(ds_meta.stats) == 1:
        # Single robot type - use its stats directly
        robot_type = list(ds_meta.stats.keys())[0]
        kwargs["dataset_stats"] = ds_meta.stats[robot_type]
    elif ds_meta.stats and len(ds_meta.stats) > 1:
        # Multiple robot types - aggregate statistics across all robot types
        aggregated_stats = {}
        for robot_type, stats in ds_meta.stats.items():
            for feature_name, feature_stats in stats.items():
                if feature_name not in aggregated_stats:
                    aggregated_stats[feature_name] = feature_stats
                else:
                    # Use stats with larger variance to be more conservative
                    existing = aggregated_stats[feature_name]
                    if 'std' in feature_stats and 'std' in existing:
                        if feature_stats['std'].mean() > existing['std'].mean():
                            aggregated_stats[feature_name] = feature_stats
        kwargs["dataset_stats"] = aggregated_stats
    else:
        kwargs["dataset_stats"] = ds_meta.stats

    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    kwargs["config"] = cfg

    if cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    return policy
