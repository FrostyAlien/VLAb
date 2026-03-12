#!/usr/bin/env python3
import os
import random

import pytest

from lerobot.datasets.hf_streaming_dataset import (
    HFStreamingMultiLeRobotDataset,
    buffer_shuffle,
    resolve_delta_positions,
    weighted_interleave,
)


def test_resolve_delta_positions_boundary_padding():
    deltas = [-2, -1, 0, 1, 3]

    # Start of episode
    positions, is_pad = resolve_delta_positions(anchor_idx=0, episode_size=4, deltas=deltas)
    assert positions == [0, 0, 0, 1, 3]
    assert is_pad == [True, True, False, False, False]

    # End of episode
    positions, is_pad = resolve_delta_positions(anchor_idx=3, episode_size=4, deltas=deltas)
    assert positions == [1, 2, 3, 3, 3]
    assert is_pad == [False, False, False, True, True]


def test_weighted_interleave_and_buffer_shuffle_are_seeded():
    weights = [0.6, 0.3, 0.1]

    def run_once(seed: int):
        stream_a = iter([{"v": "a0"}, {"v": "a1"}, {"v": "a2"}])
        stream_b = iter([{"v": "b0"}, {"v": "b1"}, {"v": "b2"}])
        stream_c = iter([{"v": "c0"}, {"v": "c1"}, {"v": "c2"}])
        rng = random.Random(seed)
        merged = weighted_interleave([stream_a, stream_b, stream_c], weights, rng)
        return [row["v"] for row in buffer_shuffle(merged, buffer_size=3, rng=rng)]

    first = run_once(1234)
    second = run_once(1234)

    assert first == second
    assert len(first) == 9


@pytest.mark.skipif(
    not (os.getenv("HF_STREAMING_SMOKE_ROOT") and os.getenv("HF_STREAMING_SMOKE_REPO_ID")),
    reason="Set HF_STREAMING_SMOKE_ROOT and HF_STREAMING_SMOKE_REPO_ID to run local smoke test.",
)
def test_hf_streaming_smoke_local_cache():
    root = os.environ["HF_STREAMING_SMOKE_ROOT"]
    repo_id = os.environ["HF_STREAMING_SMOKE_REPO_ID"]

    dataset = HFStreamingMultiLeRobotDataset(
        repo_ids=[repo_id],
        root=root,
        shuffle=True,
        shuffle_buffer_size=128,
        num_shards=1,
        partition_by_worker=True,
    )
    iterator = iter(dataset)
    sample = next(iterator)

    assert isinstance(sample, dict)
    assert "dataset_index" in sample
