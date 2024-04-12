# Gasper-Siesta -- The Beacon Chain

## Table of contents
<!-- TOC -->
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Introduction](#introduction)
- [Constants](#constants)
  - [Misc](#misc)
- [Preset](#preset)
  - [State list lengths](#state-list-lengths)
- [Containers](#containers)
  - [Misc dependencies](#misc-dependencies)
    - [`ChainHistory`](#chainhistory)
  - [Beacon operations](#beacon-operations)
    - [`Attestation`](#attestation)
  - [Beacon state](#beacon-state)
    - [`BeaconState`](#beaconstate)
- [Helper functions](#helper-functions)
  - [Epoch processing](#epoch-processing)
    - [Justification and Finalization](#justification-and-finalization)
      - [Helpers](#helpers)
  - [Block processing](#block-processing)
    - [Operations](#operations)
      - [Attestations](#attestations)
- [Testing](#testing)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->
<!-- /TOC -->



## Introduction
Gasper-Siesta aims to reduce commit latency in the Beacon Chain by modifying the Casper FFG finality rule.


## Constants

### Misc 

| Name | Value |
| - | - |
| `JUSTIFICATION_BITS_LENGTH` | `uint64(256)` |


## Preset

### State list lengths 
| Name | Value | Unit | Duration |
| - | - | :-: | :-: |
| `HISTORICAL_EPOCH_FINALITY_WINDOW` | `uint64(2**8)` (= 256) | epochs | ~27 hours |


## Containers

### Misc dependencies

#### `ChainHistory`
```python
class ChainHistory(Container):
    block_root: Root
    parent_root: Root
    slot: Slot
    parent_slot: Slot
```
NOTE: We consider a `ChainHistory` object to be empty if the `block_root` or `parent_root` is the zero hash.


### Beacon operations

#### `Attestation`

```python
class Attestation(phase0.Attestation):
    # Store the chain of block roots that were used to construct this attestation
    justification_chain: Vector[ChainHistory, SLOTS_PER_EPOCH * HISTORICAL_EPOCH_FINALITY_WINDOW]
```


### Beacon state

#### `BeaconState`

```python
class BeaconState(phase0.BeaconState):
    # Store historical `Attestation` across all slots
    historical_attestations: Vector[List[Attestation, MAX_ATTESTATIONS], SLOTS_PER_EPOCH * HISTORICAL_EPOCH_FINALITY_WINDOW]
    # Store a `ChainHistory` container per slot for each epoch in our finality window
    historical_chain: Vector[ChainHistory, SLOTS_PER_EPOCH * HISTORICAL_EPOCH_FINALITY_WINDOW]
```

## Helper functions

### Epoch processing

#### Justification and Finalization
```python
def weigh_justification_and_finalization(state: BeaconState,
                                         total_active_balance: Gwei,
                                         previous_epoch_target_balance: Gwei,
                                         current_epoch_target_balance: Gwei) -> None:
    previous_epoch = get_previous_epoch(state)
    current_epoch = get_current_epoch(state)
    old_previous_justified_checkpoint = state.previous_justified_checkpoint
    old_current_justified_checkpoint = state.current_justified_checkpoint

    # Process justifications
    state.previous_justified_checkpoint = state.current_justified_checkpoint
    state.justification_bits[1:] = state.justification_bits[:JUSTIFICATION_BITS_LENGTH - 1]
    state.justification_bits[0] = 0b0
    if previous_epoch_target_balance * 3 >= total_active_balance * 2:
        state.current_justified_checkpoint = Checkpoint(epoch=previous_epoch,
                                                        root=get_block_root(state, previous_epoch))
        state.justification_bits[1] = 0b1
    if current_epoch_target_balance * 3 >= total_active_balance * 2:
        state.current_justified_checkpoint = Checkpoint(epoch=current_epoch,
                                                        root=get_block_root(state, current_epoch))
        state.justification_bits[0] = 0b1

    for epoch in range(state.previous_justified_checkpoint.epoch, current_epoch):
        previous_block_root = state.historical_block_roots[epoch % SLOTS_PER_HISTORICAL_ROOT]
        conflicting_stake = get_conflicting_historical_attestation_stake(state, epoch, previous_block_root)
        # TODO: check if threshold correct
        if conflicting_stake > Gwei(0):
            break
    state.finalized_checkpoint = state.previous_justified_checkpoint
```

##### Helpers

```python
def is_in_justified_checkpoint_chain(state: BeaconState, attestation: Attestation) -> bool:
    return all(val_history == attestation_history for val_history, attestation_history in zip(state.historical_chain, attestation.justification_chain))
```

```python
def get_conflicting_historical_attestation_stake(state: BeaconState, epoch: Epoch, block_root: Root) -> Gwei:
    """
    Return the total stake of validators that made conflicting attestations for the given epoch and block root.
    """ 
    conflicting_stake = Gwei(0)
    for slot_attestations in state.historical_attestations:
        for attestation in slot_attestations:
            if attestation.data.target.epoch == epoch and not is_in_justified_checkpoint_chain(state, attestation):
                # TODO: Check if we can check historical effective balance
                conflicting_stake += sum(state.validators[index].effective_balance for index in get_attesting_indices(state, attestation))
    return conflicting_stake
```

```python
def get_total_active_balance_at_epoch(state: BeaconState, epoch: Epoch) -> Gwei:
    # TODO: Check if we can check historical effective balance
    return Gwei(sum(
        state.validators[index].effective_balance for index in get_active_validator_indices(state, epoch)
    ))
```

### Block processing

#### Operations

##### Attestations

```python
def process_attestation(state: BeaconState, attestation: Attestation) -> None:
    data = attestation.data
    assert data.target.epoch in (get_previous_epoch(state), get_current_epoch(state))
    assert data.target.epoch == compute_epoch_at_slot(data.slot)
    assert data.slot + MIN_ATTESTATION_INCLUSION_DELAY <= state.slot <= data.slot + SLOTS_PER_EPOCH
    assert data.index < get_committee_count_per_slot(state, data.target.epoch)

    committee = get_beacon_committee(state, data.slot, data.index)
    assert len(attestation.aggregation_bits) == len(committee)

    pending_attestation = PendingAttestation(
        data=data,
        aggregation_bits=attestation.aggregation_bits,
        inclusion_delay=state.slot - data.slot,
        proposer_index=get_beacon_proposer_index(state),
    )

    if data.target.epoch == get_current_epoch(state):
        assert data.source == state.current_justified_checkpoint
        state.current_epoch_attestations.append(pending_attestation)
        assert len(state.historical_epoch_attestations[0]) < HISTORICAL_EPOCH_FINALITY_WINDOW
        state.historical_epoch_attestations[0].append(attestation)
    else:
        assert data.source == state.previous_justified_checkpoint
        state.previous_epoch_attestations.append(pending_attestation)
        assert len(state.historical_epoch_attestations[1]) < HISTORICAL_EPOCH_FINALITY_WINDOW
        state.historical_epoch_attestations[1].append(attestation)

    # Verify signature
    assert is_valid_indexed_attestation(state, get_indexed_attestation(state, attestation))
```

## Testing

```python
def initialize_beacon_state_from_eth1(eth1_block_hash: Hash32,
                                      eth1_timestamp: uint64,
                                      deposits: Sequence[Deposit]
                                      ) -> BeaconState:
    state_phase0 = phase0.initialize_beacon_state_from_eth1(
        eth1_block_hash,
        eth1_timestamp,
        deposits,
    )
    state = upgrade_to_gasper_siesta(state_phase0)
    state.fork.previous_version = GASPER_SIESTA_FORK_VERSION
    state.fork.current_version = GASPER_SIESTA_FORK_VERSION
    return state
```