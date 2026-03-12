## ADDED Requirements

### Requirement: Hugging Face Native Streaming Backend
The training pipeline SHALL support a dataset backend named `hf_streaming` that streams samples directly from locally available Hugging Face datasets without converting data to another storage format.

#### Scenario: Select hf_streaming backend
- **GIVEN** `dataset.backend` is set to `hf_streaming`
- **AND** local dataset roots for configured repo IDs are available
- **WHEN** `make_dataset()` is called by training setup
- **THEN** the returned dataset object uses iterable streaming semantics
- **AND** no conversion step is required before training starts

#### Scenario: Missing local dataset root
- **GIVEN** `dataset.backend` is `hf_streaming`
- **AND** one or more configured repo roots are missing
- **WHEN** dataset initialization runs
- **THEN** initialization SHALL fail fast with an actionable error that identifies missing repo roots

### Requirement: Approximate Shuffle and Weighted Multi-Repo Mixing
The `hf_streaming` backend SHALL support weighted interleaving across multiple repos and approximate shuffle via a bounded buffer with deterministic seed control.

#### Scenario: Weighted interleave with deterministic seed
- **GIVEN** three repos and explicit sampling weights
- **AND** a fixed shuffle seed and epoch index
- **WHEN** a training epoch starts
- **THEN** the backend SHALL produce a mixed stream that reflects configured weight proportions over time
- **AND** repeated runs with identical inputs SHALL reproduce the same stream order for that epoch

#### Scenario: Approximate shuffle enabled
- **GIVEN** shuffle buffer size greater than 1
- **WHEN** the backend iterates samples
- **THEN** output order SHALL differ from strict per-repo sequential order
- **AND** memory usage SHALL be bounded by configured buffer and worker settings

### Requirement: Streaming Temporal Delta Semantics
For policies requiring temporal delta features, the `hf_streaming` backend SHALL construct delta-aligned observations/actions/rewards from a streaming window and emit padding indicators for out-of-range positions.

#### Scenario: Positive and negative delta offsets inside an episode
- **GIVEN** configured delta offsets that include negative and positive values
- **WHEN** an anchor frame is emitted from an episode
- **THEN** the backend SHALL populate delta fields using buffered frames aligned to those offsets
- **AND** output keys SHALL match the existing training data contract

#### Scenario: Episode boundary padding
- **GIVEN** an anchor frame near episode start or end
- **WHEN** requested delta offsets extend outside the episode bounds
- **THEN** the backend SHALL clamp/pad out-of-range slots
- **AND** corresponding `*_is_pad` flags SHALL be set consistently

### Requirement: Train DataLoader Compatibility for Iterable Backends
The offline training pipeline SHALL configure DataLoader behavior for iterable backends without index-based samplers.

#### Scenario: Build dataloader for hf_streaming backend
- **GIVEN** `dataset.backend=hf_streaming`
- **WHEN** `train.py` constructs the DataLoader
- **THEN** index-based sampler paths SHALL be disabled for this backend
- **AND** DataLoader-level random shuffle SHALL be disabled when backend-internal shuffle is active
- **AND** collate/padding behavior SHALL remain compatible with existing policy inputs

### Requirement: Streaming Resume and Epoch Progression
The training pipeline SHALL provide deterministic epoch progression for `hf_streaming` and define resume behavior for stream position state.

#### Scenario: Resume from checkpoint
- **GIVEN** a checkpoint saved during training with `hf_streaming`
- **WHEN** training resumes with the same config and seed
- **THEN** epoch-level shuffle determinism SHALL be preserved
- **AND** stream-position resume behavior SHALL be explicit (exact position when available, otherwise deterministic next-epoch fallback)
