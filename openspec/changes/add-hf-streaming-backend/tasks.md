## 1. Backend Scaffold
- [x] 1.1 Add `hf_streaming` backend option to dataset config and factory routing.
- [x] 1.2 Add a new iterable dataset backend module for local HF datasets (single repo + multi-repo).
- [x] 1.3 Add backend-specific config fields (shuffle buffer, shard count, epoch seed behavior, optional worker partition mode).

## 2. Streaming Semantics
- [x] 2.1 Implement repo interleave with sampling weights and local-path validation.
- [x] 2.2 Implement approximate shuffle pipeline with deterministic seeding.
- [x] 2.3 Implement temporal delta window builder (negative/positive offsets + `_is_pad` parity).

## 3. Train Pipeline Integration
- [x] 3.1 Update train dataloader path so iterable backends bypass index sampler and DataLoader-level shuffle.
- [x] 3.2 Keep existing collate/padding contract unchanged for policy compatibility.
- [x] 3.3 Add resume handling for iterable backend (epoch and state behavior).

## 4. Validation
- [x] 4.1 Add unit tests for window/padding correctness at episode boundaries.
- [x] 4.2 Add integration tests for weighted interleave + seeded shuffle reproducibility.
- [x] 4.3 Add a short offline training smoke test using local HF cache and `backend=hf_streaming`.

## 5. Documentation
- [x] 5.1 Add usage examples for local data streaming, recommended defaults, and known trade-offs (approximate shuffle).

## Dependencies / Parallelism
- 1.1-1.3 must complete before sections 2 and 3.
- 2.1 and 2.2 can run in parallel.
- 2.3 depends on 2.1 data iteration shape.
- Section 4 depends on sections 2 and 3.
