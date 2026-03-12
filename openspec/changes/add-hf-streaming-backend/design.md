## Context
The current offline training path is index-oriented:
- `LeRobotDataset.__getitem__(idx)` retrieves a base row with `hf_dataset[idx]`.
- Temporal queries use index lists (`select(query_indices)`) for delta features.
- `EpisodeAwareSampler` and DataLoader sampler/shuffle assume map-style random indexing.

This does not scale for very large multi-repo corpora when all map-style datasets are kept open or when conversion to MDS is operationally expensive.

## Goals / Non-Goals
- Goals:
  - Stream directly from local Hugging Face dataset storage.
  - Keep memory bounded and independent of total corpus size.
  - Preserve output sample schema expected by training/policy code.
  - Support approximate shuffle across many repos.
- Non-Goals:
  - Bitwise-equivalent ordering to current indexed sampling.
  - Replacement of the existing `hf` and `mosaic` backends.

## Proposed Architecture
### 1) New backend type
Add `dataset.backend=hf_streaming` in config and `make_dataset()` factory branch.

### 2) Iterable multi-repo dataset
Create a new iterable dataset module (e.g., `hf_streaming_dataset.py`) that:
- Opens each repo from local disk (`load_from_disk` / local split).
- Converts map-style split to iterable (`to_iterable_dataset(num_shards=...)`).
- Interleaves repo iterables using configured weights.
- Applies buffer shuffle (`shuffle(buffer_size=...)`) and epoch reseeding.

### 3) Streaming temporal window builder
Replace indexed delta lookup with streaming window logic:
- Maintain per-stream episode-aware ring buffer.
- Hold lookahead up to `max_positive_delta` before emitting anchor frame.
- Resolve negative/positive deltas from buffered frames.
- Emit `_is_pad` flags for out-of-range positions and episode boundaries.

### 4) Train loop integration
For iterable backends:
- Disable sampler-based indexing (`EpisodeAwareSampler` not used).
- Disable DataLoader `shuffle=True` (shuffle handled by dataset pipeline).
- Keep collate/padding behavior unchanged.

### 5) Resume semantics
- Use epoch-based reshuffle (`set_epoch(epoch)` equivalent strategy).
- Support checkpointing iterable position when backend/runtime permits; otherwise resume at next epoch boundary with deterministic seed progression.

## Key Decisions
- Decision: Use HF iterable APIs directly instead of conversion.
  - Rationale: avoids conversion/storage overhead and reuses existing local data layout.
- Decision: Accept approximate shuffle.
  - Rationale: required for scalable streaming; exact global-random requires full index materialization.
- Decision: Preserve sample schema compatibility.
  - Rationale: minimizes policy and collate changes.

## Alternatives Considered
- Continue with Mosaic conversion:
  - Pros: mature streaming/shuffle features.
  - Cons: conversion latency, storage amplification, operational complexity.
- HF map-style + LRU dataset handles:
  - Pros: retains exact index semantics.
  - Cons: severe repo thrashing with global-random index sampling.

## Risks / Trade-offs
- Approximate shuffle can alter training dynamics.
  - Mitigation: configurable buffer size and deterministic seed controls.
- Temporal-window implementation complexity for positive deltas.
  - Mitigation: explicit tests for boundary/padding parity.
- Multi-worker duplication risk for iterable datasets.
  - Mitigation: sharded iterable construction and worker-aware partitioning tests.

## Validation Strategy
- Unit tests for temporal window generation and pad flags against small synthetic episodes.
- Integration test for multi-repo interleave + shuffle determinism (seeded).
- Smoke training test: one short run with `backend=hf_streaming` and local datasets.

## Open Questions
- Should resume require exact sample-position restoration, or is deterministic epoch-level resume sufficient?
- What default shuffle buffer size balances randomness and throughput for 400+ repos?
