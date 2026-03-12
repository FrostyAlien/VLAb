# Change: Add Hugging Face Native Streaming Backend

## Why
Current large-scale training (400+ repos, ~200GB local cache) needs streaming to avoid RAM pressure, but Mosaic conversion is too slow and increases storage overhead. The project already stores data in Hugging Face dataset format, so we need a backend that streams directly from local HF datasets.

## What Changes
- Add a new dataset backend (`dataset.backend=hf_streaming`) that reads from locally downloaded HF datasets without MDS conversion.
- Implement iterable-style multi-repo loading with approximate shuffle using Hugging Face iterable datasets.
- Preserve current training data contract (feature keys, image transforms, padding/collation behavior) while changing the data access pattern from random-index to streaming iteration.
- Update training data loader setup so iterable backends do not use index-based sampler/shuffle paths.
- Add resume-aware stream state handling (epoch-aware shuffle and dataloader state save/restore where supported).
- Add documentation and config examples for local-cache streaming usage.

## Scope Boundaries
- In scope: offline training backend path, dataset factory wiring, iterable dataset implementation, config surface, tests for backend behavior.
- Out of scope: Mosaic conversion pipeline, online RL buffer changes, model architecture changes.
- Non-goal: exact global-random sampling equivalence with map-style indexed sampling.

## Impact
- Affected specs: `hf-streaming-backend`
- Affected code (planned):
  - `src/lerobot/datasets/factory.py`
  - `src/lerobot/datasets/` (new iterable backend module)
  - `src/lerobot/scripts/train.py`
  - `src/lerobot/configs/default.py`
  - tests under `tests/`

## Risks
- Temporal-window logic currently depends on indexed access (`__getitem__`, `select(query_indices)`), so streaming requires window buffering logic changes.
- Approximate shuffle may change convergence characteristics versus exact global index shuffling.
- Multi-worker iterable behavior must avoid duplicate sample emission.

## Assumptions
- Data is already present locally (e.g., under HF cache root) and no remote streaming is required.
- Approximate shuffle is acceptable.
- Existing feature mapping and transforms must remain compatible with current policies.
