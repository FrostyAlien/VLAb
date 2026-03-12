#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

import logging
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from datasets import load_dataset
from datasets.utils import logging as hf_datasets_logging
from torchvision import transforms

from lerobot.constants import ACTION, HF_LEROBOT_HOME, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGE_2, OBS_IMAGE_3, OBS_IMAGE_4, OBS_STATE
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import get_delta_indices
from lerobot.datasets.utils_must import (
    ROBOT_TYPE_KEYS_MAPPING,
    create_padded_features,
    map_dict_keys,
    pad_tensor,
    reshape_features_to_max_dim,
    str_to_torch_dtype,
)
from lerobot.datasets.video_utils import decode_video_frames, get_safe_default_codec


def _as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    if isinstance(value, np.generic):
        return int(value)
    return int(value)


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    if isinstance(value, np.generic):
        return float(value)
    return float(value)


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            value = value.astype(np.uint8, copy=False)
        if not value.flags.writeable:
            value = value.copy()
        return torch.from_numpy(value)
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.integer):
            return torch.tensor(int(value), dtype=torch.int64)
        if np.issubdtype(value.dtype, np.floating):
            return torch.tensor(float(value), dtype=torch.float32)
        if np.issubdtype(value.dtype, np.bool_):
            return torch.tensor(bool(value), dtype=torch.bool)
    if isinstance(value, bool):
        return torch.tensor(value, dtype=torch.bool)
    if isinstance(value, int):
        return torch.tensor(value, dtype=torch.int64)
    if isinstance(value, float):
        return torch.tensor(value, dtype=torch.float32)
    return torch.as_tensor(value)


def _convert_raw_row_value(value: Any, to_tensor_transform: Callable) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, PIL.Image.Image):
        return to_tensor_transform(value)
    if isinstance(value, list):
        try:
            return torch.tensor(value)
        except Exception:
            return value
    if isinstance(value, tuple):
        try:
            return torch.tensor(value)
        except Exception:
            return value
    if isinstance(value, (bool, int, float, np.ndarray, np.generic, torch.Tensor)):
        return _to_tensor(value)
    return value


def resolve_delta_positions(anchor_idx: int, episode_size: int, deltas: list[int]) -> tuple[list[int], list[bool]]:
    """Compute clamped indices and padding flags for one anchor frame."""
    positions = []
    is_pad = []
    max_idx = episode_size - 1
    for delta in deltas:
        raw_idx = anchor_idx + delta
        clamped_idx = min(max(raw_idx, 0), max_idx)
        positions.append(clamped_idx)
        is_pad.append(raw_idx < 0 or raw_idx > max_idx)
    return positions, is_pad


def weighted_interleave(
    iterators: list[Iterator[dict[str, Any]]],
    weights: list[float],
    rng: random.Random,
) -> Iterator[dict[str, Any]]:
    active_iters = list(iterators)
    active_weights = [float(w) for w in weights]
    while active_iters:
        choice = rng.choices(range(len(active_iters)), weights=active_weights, k=1)[0]
        try:
            yield next(active_iters[choice])
        except StopIteration:
            del active_iters[choice]
            del active_weights[choice]


def buffer_shuffle(
    source: Iterator[dict[str, Any]], buffer_size: int, rng: random.Random
) -> Iterator[dict[str, Any]]:
    if buffer_size <= 1:
        yield from source
        return

    buffer: list[dict[str, Any]] = []
    for item in source:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        pick = rng.randrange(len(buffer))
        yield buffer[pick]
        buffer[pick] = item

    while buffer:
        pick = rng.randrange(len(buffer))
        yield buffer.pop(pick)


_CAMERA_KEY_PRIORITY = {
    OBS_IMAGE: 0,
    OBS_IMAGE_2: 1,
    OBS_IMAGE_3: 2,
    OBS_IMAGE_4: 3,
}


def _camera_sort_key(key: str) -> tuple[int, str]:
    return (_CAMERA_KEY_PRIORITY.get(key, len(_CAMERA_KEY_PRIORITY)), key)


class HFStreamingMultiLeRobotDatasetMeta:
    def __init__(
        self,
        dataset_metas: list[LeRobotDatasetMetadata],
        repo_ids: list[str],
        keys_to_max_dim: dict[str, int | None],
        train_on_all_features: bool = False,
        max_num_images: int | None = None,
    ):
        self.repo_ids = repo_ids
        self.keys_to_max_dim = keys_to_max_dim
        self.train_on_all_features = train_on_all_features

        for repo_id, ds_meta in zip(repo_ids, dataset_metas, strict=True):
            ds_meta.info["robot_type"] = ROBOT_TYPE_KEYS_MAPPING.get(repo_id, ds_meta.info["robot_type"])

        self.disabled_features = set()
        if not self.train_on_all_features:
            intersection = set(dataset_metas[0].features)
            for ds_meta in dataset_metas:
                intersection.intersection_update(ds_meta.features)
            if not intersection:
                raise RuntimeError("No common features across datasets.")
            for repo_id, ds_meta in zip(repo_ids, dataset_metas, strict=True):
                extra = set(ds_meta.features) - intersection
                if extra:
                    logging.warning(f"Disabling {extra} for repo {repo_id}")
                self.disabled_features.update(extra)

        if max_num_images is not None:
            max_num_images = int(max_num_images)
            if max_num_images < 0:
                raise ValueError(f"max_num_images must be >= 0. Got {max_num_images}.")

            candidate_camera_keys = sorted(
                {
                    key
                    for ds_meta in dataset_metas
                    for key, feature in ds_meta.features.items()
                    if key not in self.disabled_features
                    and isinstance(feature, dict)
                    and feature.get("dtype") in ["video", "image"]
                },
                key=_camera_sort_key,
            )
            dropped_camera_keys = set(candidate_camera_keys[max_num_images:])
            if dropped_camera_keys:
                logging.info(
                    "Applying max_num_images=%s. Keeping camera keys: %s. Dropping: %s",
                    max_num_images,
                    candidate_camera_keys[:max_num_images],
                    sorted(dropped_camera_keys),
                )
                self.disabled_features.update(dropped_camera_keys)

        self.union_features = {}
        for ds_meta in dataset_metas:
            for key, value in ds_meta.features.items():
                if key not in self.disabled_features:
                    self.union_features[key] = value

        self.features = reshape_features_to_max_dim(
            self.union_features,
            reshape_dim=-1,
            keys_to_max_dim=self.keys_to_max_dim,
        )

        stats_by_robot_type: dict[str, list[dict[str, dict[str, np.ndarray]]]] = {}
        for repo_id, ds_meta in zip(repo_ids, dataset_metas, strict=True):
            robot_type = ROBOT_TYPE_KEYS_MAPPING.get(repo_id, ds_meta.info["robot_type"])
            mapped_stats = {}
            if ds_meta.stats:
                mapped_stats = map_dict_keys(
                    ds_meta.stats,
                    feature_keys_mapping=ds_meta.feature_keys_mapping,
                )
                mapped_stats = {
                    key: value for key, value in mapped_stats.items() if key not in self.disabled_features
                }
            stats_by_robot_type.setdefault(robot_type, []).append(mapped_stats)

        self.stats = {}
        for robot_type, stats_list in stats_by_robot_type.items():
            valid_stats = [stats for stats in stats_list if stats]
            if not valid_stats:
                self.stats[robot_type] = {}
                continue

            try:
                self.stats[robot_type] = aggregate_stats(valid_stats)
            except ValueError as exc:
                logging.warning(
                    f"Failed to aggregate stats for robot type '{robot_type}' due to shape mismatch: {exc}. "
                    "Falling back to first available stats entry."
                )
                self.stats[robot_type] = valid_stats[0]

        for robot_type, stats_ in self.stats.items():
            for feat_key in [ACTION, OBS_ENV_STATE, OBS_STATE]:
                if feat_key not in stats_:
                    continue
                max_size = self.keys_to_max_dim.get(feat_key)
                if max_size is None:
                    continue
                for stat_name, value in stats_[feat_key].items():
                    pad_value = 0 if stat_name in ["min", "mean"] else 1
                    stats_[feat_key][stat_name] = pad_tensor(
                        value,
                        max_size=max_size,
                        pad_dim=-1,
                        pad_value=pad_value,
                    )

        self.episodes = {repo_id: ds_meta.episodes for repo_id, ds_meta in zip(repo_ids, dataset_metas, strict=True)}
        self.tasks = {repo_id: ds_meta.tasks for repo_id, ds_meta in zip(repo_ids, dataset_metas, strict=True)}
        self.info = {}
        for repo_id, ds_meta in zip(repo_ids, dataset_metas, strict=True):
            info = dict(ds_meta.info)
            info["robot_type"] = ROBOT_TYPE_KEYS_MAPPING.get(repo_id, info["robot_type"])
            self.info[repo_id] = info


@dataclass
class _RepoStreamContext:
    dataset_index: int
    repo_id: str
    root: Path
    data_dir: Path
    meta: LeRobotDatasetMetadata
    weight: float
    feature_keys_mapping: dict[str, str]
    inverse_feature_keys_mapping: dict[str, str]
    delta_indices: dict[str, list[int]] | None
    allowed_episodes: set[int] | None
    robot_type: str
    video_keys: list[str]


class HFStreamingMultiLeRobotDataset(torch.utils.data.IterableDataset):
    """LeRobot multidataset backend that streams directly from local HF datasets."""

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        split: str = "train",
        image_transforms: Callable | None = None,
        sampling_weights: list[float] | None = None,
        feature_keys_mapping: dict[str, dict[str, str]] | None = None,
        max_action_dim: int | None = None,
        max_state_dim: int | None = None,
        max_num_images: int | None = None,
        max_image_dim: int | None = None,
        train_on_all_features: bool = True,
        min_fps: int = 1,
        max_fps: int = 100,
        delta_timestamps: dict[str, dict[str, list[float]] | None] | None = None,
        episodes: dict[str, list[int] | None] | None = None,
        video_backend: str | None = None,
        tolerance_s: float = 1e-4,
        shuffle: bool = True,
        shuffle_buffer_size: int = 10_000,
        shuffle_seed: int = 9176,
        num_shards: int = 1,
        partition_by_worker: bool = True,
        max_open_repos: int = 8,
        resume_mode: str = "epoch",
        training_features: list[str] | None = None,
        quiet_hf_datasets_logs: bool = True,
    ):
        super().__init__()
        self.root = Path(root) if root else HF_LEROBOT_HOME
        self.split = split
        self.max_image_dim = max_image_dim
        self.max_num_images = max_num_images
        self.image_transforms = image_transforms
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.tolerance_s = tolerance_s
        self.shuffle = shuffle
        self.shuffle_buffer_size = max(1, int(shuffle_buffer_size))
        self.shuffle_seed = int(shuffle_seed)
        self.num_shards = max(1, int(num_shards))
        self.partition_by_worker = bool(partition_by_worker)
        self.max_open_repos = max(1, int(max_open_repos))
        self.resume_mode = resume_mode
        self.training_features = training_features
        self.quiet_hf_datasets_logs = quiet_hf_datasets_logs
        self.uses_internal_shuffle = True
        self.uses_iterable_dataset = True
        self.supports_epoch_resume = True
        self._epoch_offset = 0
        self._iter_calls = 0
        self._to_tensor_transform = transforms.ToTensor()

        sampling_weights = sampling_weights if sampling_weights is not None else [1.0] * len(repo_ids)
        if len(sampling_weights) != len(repo_ids):
            raise ValueError(
                f"The number of sampling weights must match the number of datasets. "
                f"Got {len(sampling_weights)} weights for {len(repo_ids)} datasets."
            )

        feature_keys_mapping = feature_keys_mapping or {}
        delta_timestamps = delta_timestamps or {}
        episodes = episodes or {}

        missing_roots = []
        repo_contexts: list[_RepoStreamContext] = []
        kept_repo_ids = []
        kept_metas = []

        for repo_id, weight in zip(repo_ids, sampling_weights, strict=True):
            if float(weight) <= 0:
                logging.warning(f"Skipping repo '{repo_id}' because its sampling weight is <= 0 ({weight}).")
                continue

            dataset_root = self.root / repo_id
            if not dataset_root.exists():
                missing_roots.append(str(dataset_root))
                continue
            data_dir = dataset_root / "data"
            if not data_dir.exists():
                missing_roots.append(str(data_dir))
                continue

            ds_meta = LeRobotDatasetMetadata(
                repo_id,
                root=dataset_root,
                feature_keys_mapping=feature_keys_mapping,
                local_files_only=True,
            )
            if ds_meta.fps < min_fps or ds_meta.fps > max_fps:
                logging.warning(
                    f"Skipping repo '{repo_id}' due to invalid fps={ds_meta.fps}. Allowed range: [{min_fps}, {max_fps}]"
                )
                continue

            mapping = feature_keys_mapping.get(repo_id, {}) if feature_keys_mapping else {}
            inverse_mapping = {v: k for k, v in mapping.items() if v}
            allowed_episodes = episodes.get(repo_id)
            allowed_episodes = set(allowed_episodes) if allowed_episodes is not None else None
            dt = delta_timestamps.get(repo_id)
            delta_indices = get_delta_indices(dt, ds_meta.fps) if dt is not None else None
            robot_type = ROBOT_TYPE_KEYS_MAPPING.get(repo_id, ds_meta.info["robot_type"])
            resolved_video_keys = self._resolve_available_video_keys(ds_meta, dataset_root, allowed_episodes)
            declared_video_keys = [key for key in ds_meta.video_keys if self._is_valid_video_key(key)]
            if declared_video_keys and not resolved_video_keys:
                logging.warning(
                    f"Skipping repo '{repo_id}' because no local video files were found for declared keys "
                    f"{declared_video_keys}. Check local dataset completeness."
                )
                continue

            repo_contexts.append(
                _RepoStreamContext(
                    dataset_index=len(repo_contexts),
                    repo_id=repo_id,
                    root=dataset_root,
                    data_dir=data_dir,
                    meta=ds_meta,
                    weight=float(weight),
                    feature_keys_mapping=mapping,
                    inverse_feature_keys_mapping=inverse_mapping,
                    delta_indices=delta_indices,
                    allowed_episodes=allowed_episodes,
                    robot_type=robot_type,
                    video_keys=resolved_video_keys,
                )
            )
            kept_repo_ids.append(repo_id)
            kept_metas.append(ds_meta)

        if missing_roots:
            missing = "\n".join(f"- {path}" for path in sorted(set(missing_roots)))
            raise FileNotFoundError(
                "Missing local dataset roots for hf_streaming backend. "
                f"Please verify dataset.root and repo IDs.\n{missing}"
            )

        if not repo_contexts:
            raise RuntimeError("No valid HF streaming dataset repos were found.")

        self._repo_contexts = repo_contexts
        self.repo_ids = kept_repo_ids
        self.meta = HFStreamingMultiLeRobotDatasetMeta(
            dataset_metas=kept_metas,
            repo_ids=self.repo_ids,
            keys_to_max_dim={
                ACTION: max_action_dim,
                OBS_ENV_STATE: max_state_dim,
                OBS_STATE: max_state_dim,
                "observation.image": max_image_dim,
                "observation.image2": max_image_dim,
                "observation.image3": max_image_dim,
            },
            train_on_all_features=train_on_all_features,
            max_num_images=self.max_num_images,
        )
        self.disabled_features = self.meta.disabled_features
        self.stats = self.meta.stats
        allowed_camera_keys = set(self.camera_keys)
        for context in self._repo_contexts:
            context.video_keys = [
                video_key
                for video_key in context.video_keys
                if context.feature_keys_mapping.get(video_key, video_key) in allowed_camera_keys
            ]
        self._num_frames = sum(
            self._count_frames_for_repo(context.meta, context.allowed_episodes) for context in self._repo_contexts
        )
        self._num_episodes = sum(
            self._count_episodes_for_repo(context.meta, context.allowed_episodes) for context in self._repo_contexts
        )

    @staticmethod
    def _count_frames_for_repo(meta: LeRobotDatasetMetadata, allowed_episodes: set[int] | None) -> int:
        if allowed_episodes is None:
            return int(meta.total_frames)
        return int(
            sum(
                episode["length"]
                for episode_idx, episode in meta.episodes.items()
                if int(episode_idx) in allowed_episodes
            )
        )

    @staticmethod
    def _count_episodes_for_repo(meta: LeRobotDatasetMetadata, allowed_episodes: set[int] | None) -> int:
        if allowed_episodes is None:
            return int(meta.total_episodes)
        return int(sum(1 for episode_idx in meta.episodes if int(episode_idx) in allowed_episodes))

    @property
    def repo_id_to_index(self) -> dict[str, int]:
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def repo_index_to_id(self) -> dict[int, str]:
        return {i: repo_id for i, repo_id in enumerate(self.repo_ids)}

    @property
    def fps(self) -> int:
        first_repo = self.repo_ids[0]
        return int(self.meta.info[first_repo]["fps"])

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def camera_keys(self) -> list[str]:
        return [
            key
            for key, feature in self.features.items()
            if isinstance(feature, dict) and feature.get("dtype") in ["video", "image"]
        ]

    @property
    def video_frame_keys(self) -> list[str]:
        return [
            key
            for key, feature in self.features.items()
            if isinstance(feature, dict) and feature.get("dtype") == "video"
        ]

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def num_episodes(self) -> int:
        return self._num_episodes

    def set_epoch(self, epoch: int) -> None:
        self._epoch_offset = max(0, int(epoch))
        self._iter_calls = 0

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self._epoch_offset + self._iter_calls}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.set_epoch(int(state_dict.get("epoch", 0)))

    def _resolve_source_key(self, key: str, row: dict[str, Any], inverse_mapping: dict[str, str]) -> str:
        if key in row:
            return key
        source_key = inverse_mapping.get(key)
        if source_key and source_key in row:
            return source_key
        return key

    def _resolve_video_key(self, key: str, source_key: str, video_keys: list[str]) -> str | None:
        for candidate in (source_key, key):
            for variant in self._video_key_variants(candidate):
                if variant in video_keys:
                    return variant
        return None

    @staticmethod
    def _video_key_variants(key: str) -> list[str]:
        variants = [key]
        if ".images." in key:
            variants.append(key.replace(".images.", ".", 1))
        elif key.startswith("observation.") and not key.startswith("observation.images."):
            suffix = key[len("observation.") :]
            variants.append(f"observation.images.{suffix}")
        return list(dict.fromkeys(variants))

    @staticmethod
    def _is_valid_video_key(key: str | None) -> bool:
        if key is None:
            return False
        key_norm = str(key).strip().lower()
        return key_norm not in {"", "none", "null"}

    @staticmethod
    def _probe_episode_indices(meta: LeRobotDatasetMetadata, allowed_episodes: set[int] | None) -> list[int]:
        if allowed_episodes is not None:
            source = sorted(int(ep) for ep in allowed_episodes)
        else:
            source = sorted(int(ep) for ep in meta.episodes.keys())
        return source[:3] if source else [0]

    def _resolve_available_video_keys(
        self,
        meta: LeRobotDatasetMetadata,
        dataset_root: Path,
        allowed_episodes: set[int] | None,
    ) -> list[str]:
        declared_video_keys = [key for key in meta.video_keys if self._is_valid_video_key(key)]
        if not declared_video_keys:
            return []

        episodes_to_probe = self._probe_episode_indices(meta, allowed_episodes)
        resolved_keys: list[str] = []

        for key in declared_video_keys:
            matched_key = None
            for variant in self._video_key_variants(key):
                for episode_index in episodes_to_probe:
                    video_path = dataset_root / meta.get_video_file_path(episode_index, variant)
                    if video_path.is_file():
                        matched_key = variant
                        break
                if matched_key is not None:
                    break
            if matched_key is not None and matched_key not in resolved_keys:
                resolved_keys.append(matched_key)

        return resolved_keys

    def _resize_with_pad_to_square(self, img: torch.Tensor, target_dim: int) -> torch.Tensor:
        if img.ndim not in (3, 4):
            return img

        squeeze_batch_dim = img.ndim == 3
        img_batched = img.unsqueeze(0) if squeeze_batch_dim else img
        _, _, cur_h, cur_w = img_batched.shape
        if cur_h == target_dim and cur_w == target_dim:
            return img

        ratio = max(cur_w / target_dim, cur_h / target_dim)
        resized_h = max(1, int(round(cur_h / ratio)))
        resized_w = max(1, int(round(cur_w / ratio)))
        img_resized = F.interpolate(
            img_batched,
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        )

        pad_h = max(0, target_dim - resized_h)
        pad_w = max(0, target_dim - resized_w)
        if pad_h > 0 or pad_w > 0:
            img_resized = F.pad(img_resized, (pad_w, 0, pad_h, 0), value=0.0)

        return img_resized.squeeze(0) if squeeze_batch_dim else img_resized

    def _resize_item_images(self, item: dict[str, Any]) -> dict[str, Any]:
        if self.max_image_dim is None:
            return item
        for cam_key in self.camera_keys:
            if cam_key in item and isinstance(item[cam_key], torch.Tensor):
                item[cam_key] = self._resize_with_pad_to_square(item[cam_key], self.max_image_dim)
        return item

    def _coerce_item_tensor_dtypes(self, item: dict[str, Any]) -> dict[str, Any]:
        """Force tensor dtypes to match feature schema to avoid collate dtype conflicts."""
        for key, feature in self.meta.features.items():
            if key not in item:
                continue
            value = item[key]
            if not isinstance(value, torch.Tensor):
                continue
            target_dtype = str_to_torch_dtype(feature["dtype"])
            if value.dtype != target_dtype:
                item[key] = value.to(dtype=target_dtype)
        return item

    @contextmanager
    def _quiet_hf_dataset_loading(self):
        """Temporarily silence HF Datasets progress/log spam during load_dataset()."""
        if not self.quiet_hf_datasets_logs:
            yield
            return

        prev_progress_enabled = hf_datasets_logging.is_progress_bar_enabled()
        prev_verbosity = hf_datasets_logging.get_verbosity()
        hf_datasets_logging.disable_progress_bar()
        hf_datasets_logging.set_verbosity_error()
        try:
            yield
        finally:
            if prev_progress_enabled:
                hf_datasets_logging.enable_progress_bar()
            else:
                hf_datasets_logging.disable_progress_bar()
            hf_datasets_logging.set_verbosity(prev_verbosity)

    def _iter_repo_rows(self, context: _RepoStreamContext) -> Iterator[dict[str, Any]]:
        with self._quiet_hf_dataset_loading():
            dataset = load_dataset("parquet", data_dir=str(context.data_dir), split=self.split)
        iterable_dataset = dataset.to_iterable_dataset(num_shards=self.num_shards)
        try:
            for raw_row in iterable_dataset:
                row = {
                    key: _convert_raw_row_value(value, self._to_tensor_transform)
                    for key, value in raw_row.items()
                }
                if context.allowed_episodes is not None:
                    episode_index = _as_int(row["episode_index"])
                    if episode_index not in context.allowed_episodes:
                        continue
                yield row
        finally:
            # Release per-repo dataset objects as soon as this repo iterator is exhausted.
            del iterable_dataset
            del dataset

    def _iter_repo_samples(self, context: _RepoStreamContext) -> Iterator[dict[str, Any]]:
        current_episode: int | None = None
        episode_rows: list[dict[str, Any]] = []

        for row in self._iter_repo_rows(context):
            episode_index = _as_int(row["episode_index"])
            if current_episode is None:
                current_episode = episode_index

            if episode_index != current_episode:
                yield from self._emit_episode_samples(context, episode_rows)
                episode_rows = []
                current_episode = episode_index
            episode_rows.append(row)

        if episode_rows:
            yield from self._emit_episode_samples(context, episode_rows)

    def _emit_episode_samples(
        self,
        context: _RepoStreamContext,
        episode_rows: list[dict[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        if not episode_rows:
            return

        episode_size = len(episode_rows)
        video_keys = context.video_keys

        for anchor_idx, anchor_row in enumerate(episode_rows):
            item = map_dict_keys(dict(anchor_row), feature_keys_mapping=context.feature_keys_mapping)
            query_timestamps = {
                video_key: [_as_float(anchor_row["timestamp"])] for video_key in video_keys
            }

            if context.delta_indices is not None:
                for key, deltas in context.delta_indices.items():
                    source_key = self._resolve_source_key(
                        key,
                        anchor_row,
                        context.inverse_feature_keys_mapping,
                    )
                    query_positions, is_pad = resolve_delta_positions(anchor_idx, episode_size, deltas)
                    item[f"{key}_is_pad"] = torch.BoolTensor(is_pad)

                    video_key = self._resolve_video_key(key, source_key, video_keys)
                    if video_key is not None:
                        query_timestamps[video_key] = [
                            _as_float(episode_rows[pos]["timestamp"]) for pos in query_positions
                        ]
                    else:
                        if source_key not in anchor_row:
                            continue
                        stacked_values = [_to_tensor(episode_rows[pos][source_key]) for pos in query_positions]
                        item[key] = torch.stack(stacked_values)

            if video_keys:
                episode_index = _as_int(anchor_row["episode_index"])
                for video_key, timestamps in query_timestamps.items():
                    video_path = context.root / context.meta.get_video_file_path(episode_index, video_key)
                    frames = decode_video_frames(video_path, timestamps, self.tolerance_s, self.video_backend)
                    mapped_video_key = context.feature_keys_mapping.get(video_key, video_key)
                    item[mapped_video_key] = frames.squeeze(0)

            if "task_index" in item:
                try:
                    task_index = _as_int(item["task_index"])
                    item["task"] = context.meta.tasks[task_index]
                except Exception:
                    pass

            if "robot_type" not in item:
                item["robot_type"] = context.robot_type
            item["dataset_index"] = torch.tensor(context.dataset_index, dtype=torch.int64)

            item = map_dict_keys(
                item,
                feature_keys_mapping=context.feature_keys_mapping,
                training_features=self.training_features,
            )
            item = self._coerce_item_tensor_dtypes(item)

            item = create_padded_features(item, self.meta.features)
            for data_key in self.disabled_features:
                if data_key in item:
                    del item[data_key]
                pad_mask_key = f"{data_key}_padding_mask"
                if pad_mask_key in item:
                    del item[pad_mask_key]
                is_pad_key = f"{data_key}_is_pad"
                if is_pad_key in item:
                    del item[is_pad_key]

            if self.image_transforms is not None:
                for cam_key in self.camera_keys:
                    if cam_key in item:
                        item[cam_key] = self.image_transforms(item[cam_key])

            item = self._resize_item_images(item)
            yield item

    def _contexts_for_worker(self, worker_id: int, num_workers: int) -> list[_RepoStreamContext]:
        if not self.partition_by_worker:
            return self._repo_contexts
        process_rank = 0
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            process_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            process_rank = int(os.environ.get("RANK", 0))

        total_consumers = max(1, world_size * num_workers)
        consumer_id = process_rank * num_workers + worker_id
        return [ctx for i, ctx in enumerate(self._repo_contexts) if i % total_consumers == consumer_id]

    @staticmethod
    def _distributed_process_info() -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))

    def __len__(self):
        return self.num_frames

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        process_rank, world_size = self._distributed_process_info()
        total_consumers = max(1, world_size * num_workers)
        consumer_id = process_rank * num_workers + worker_id

        contexts = self._contexts_for_worker(worker_id, num_workers)
        if not contexts:
            return iter(())

        epoch = self._epoch_offset + self._iter_calls
        self._iter_calls += 1
        seed = self.shuffle_seed + epoch * 1_000_003 + consumer_id * 10_007 + total_consumers
        rng = random.Random(seed)

        repo_weights = [context.weight for context in contexts]
        if len(contexts) <= self.max_open_repos:
            repo_iterators = [iter(self._iter_repo_samples(context)) for context in contexts]
            merged_stream = weighted_interleave(repo_iterators, repo_weights, rng)
        else:
            merged_stream = self._weighted_interleave_limited_open(contexts, rng)

        if self.shuffle:
            yield from buffer_shuffle(merged_stream, self.shuffle_buffer_size, rng)
        else:
            yield from merged_stream

    def _weighted_interleave_limited_open(
        self,
        contexts: list[_RepoStreamContext],
        rng: random.Random,
    ) -> Iterator[dict[str, Any]]:
        """Interleave repos while capping concurrently open iterators to bound RAM usage."""
        pending_indices = list(range(len(contexts)))
        active_iters: list[Iterator[dict[str, Any]]] = []
        active_weights: list[float] = []

        def activate_one() -> None:
            if not pending_indices:
                return
            pending_weights = [contexts[i].weight for i in pending_indices]
            picked_pos = rng.choices(range(len(pending_indices)), weights=pending_weights, k=1)[0]
            ctx_idx = pending_indices.pop(picked_pos)
            active_iters.append(iter(self._iter_repo_samples(contexts[ctx_idx])))
            active_weights.append(contexts[ctx_idx].weight)

        for _ in range(min(self.max_open_repos, len(pending_indices))):
            activate_one()

        while active_iters:
            choice = rng.choices(range(len(active_iters)), weights=active_weights, k=1)[0]
            try:
                yield next(active_iters[choice])
            except StopIteration:
                del active_iters[choice]
                del active_weights[choice]
                activate_one()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys},\n"
            f"  Shuffle: {self.shuffle},\n"
            f"  Shuffle buffer size: {self.shuffle_buffer_size},\n"
            f"  Max open repos: {self.max_open_repos},\n"
            f")"
        )
