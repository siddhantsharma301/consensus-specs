# `beacon-chain.md` Template

# Gasper-Siesta

## Table of contents
<!-- TOC -->
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Introduction](#introduction)
- [Notation](#notation)
- [Custom types](#custom-types)
- [Constants](#constants)
  - [Misc](#misc)
- [Preset](#preset)
  - [State list lengths](#state-list-lengths)
- [Containers](#containers)
  - [Beacon state](#beacon-state)
    - [`BeaconState`](#beaconstate)
- [Helper functions](#helper-functions)
  - [Epoch processing](#epoch-processing)
    - [Justification and Finalization](#justification-and-finalization)
      - [Helpers](#helpers)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->
<!-- /TOC -->



## Introduction
Gasper-Siesta aims to reduce commit latency in the Beacon Chain by modifying the Casper FFG finality rule.

## Notation

## Custom types

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

### Beacon state

#### `BeaconState`

```python
class BeaconState(phase0.BeaconState):
    historical_epoch_attestations: Vector[Attestation, MAX_ATTESTATIONS * HISTORICAL_EPOCH_FINALITY_WINDOW]
    historical_epoch_block_roots: Vector[Root, HISTORICAL_EPOCH_FINALITY_WINDOW]
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
def get_matching_historical_target_attestations(state: BeaconState, epoch: Epoch, block_root: Root) -> Sequence[Attestation]:
    return [
        a for a in state.historical_attestations 
        if a.data.target.root == block_root and a.data.target.epoch == epoch
    ]
```

```python
def get_conflicting_historical_attestation_stake(state: BeaconState, epoch: Epoch, block_root: Root) -> Gwei:
    """
    Return the total stake of validators that made conflicting attestations for the given epoch and block root.
    """ 
    conflict_stake = Gwei(0)
    for attestation in state.historical_attestations:
        if attestation.data.target.epoch == epoch and attestation.data.target.root != block_root:
            for index in get_attesting_indices(state, attestation):
                # TODO: Check if we can check historical effective balance
                conflicting_stake += state.validators[index].effective_balance
    return conflicting_stake
```

```python
def get_total_active_balance_at_epoch(state: BeaconState, epoch: Epoch) -> Gwei:
    # TODO: Check if we can check historical effective balance
    return Gwei(sum(
        state.validators[index].effective_balance for index in get_active_validator_indices(state, epoch)
    ))
```


<!-- 
## Testing

*Note*: The function `initialize_beacon_state_from_eth1` is modified for pure <FORK_NAME> testing only.

```python
def initialize_beacon_state_from_eth1(eth1_block_hash: Hash32,
                                      eth1_timestamp: uint64,
                                      deposits: Sequence[Deposit],
                                      execution_payload_header: ExecutionPayloadHeader=ExecutionPayloadHeader()
                                      ) -> BeaconState:
    ...
``` 
-->
