# Gasper-Siesta -- Fork Logic

## Table of contents

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Introduction](#introduction)
- [Configuration](#configuration)
- [Helper functions](#helper-functions)
  - [Misc](#misc)
    - [`compute_fork_version`](#compute_fork_version)
- [Fork to Gasper-Siesta](#fork-to-gasper-siesta)
  - [Fork trigger](#fork-trigger)
  - [Upgrading the state](#upgrading-the-state)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Introduction

This document describes the process of the first upgrade of the beacon chain: the Gasper-Siesta hard fork, introducing light client support and other improvements.

## Configuration

| Name | Value |
| - | - |
| `GASPER_SIESTA_FORK_VERSION` | `Version('0x01000000')` |
| `GASPER_SIESTA_FORK_EPOCH` | `Epoch(74240)` (Oct 27, 2021, 10:56:23am UTC) |

## Helper functions

### Misc

#### `compute_fork_version`

```python
def compute_fork_version(epoch: Epoch) -> Version:
    """
    Return the fork version at the given ``epoch``.
    """
    if epoch >= GASPER_SIESTA_FORK_EPOCH:
        return GASPER_SIESTA_FORK_VERSION
    return GENESIS_FORK_VERSION
```

## Fork to Gasper-Siesta

### Fork trigger

The fork is triggered at epoch `GASPER_SIESTA_FORK_EPOCH`.

Note that for the pure Gasper-Siesta networks, we don't apply `upgrade_to_gasper_siesta` since it starts with Gasper-Siesta version logic.

### Upgrading the state

If `state.slot % SLOTS_PER_EPOCH == 0` and `compute_epoch_at_slot(state.slot) == GASPER_SIESTA_FORK_EPOCH`, an irregular state change is made to upgrade to Gasper-Siesta.

The upgrade occurs after the completion of the inner loop of `process_slots` that sets `state.slot` equal to `GASPER_SISTA_FORK_EPOCH * SLOTS_PER_EPOCH`.
Care must be taken when transitioning through the fork boundary as implementations will need a modified [state transition function](../phase0/beacon-chain.md#beacon-chain-state-transition-function) that deviates from the Phase 0 document.
In particular, the outer `state_transition` function defined in the Phase 0 document will not expose the precise fork slot to execute the upgrade in the presence of skipped slots at the fork boundary. Instead the logic must be within `process_slots`.

```python
def populate_historical_attestations(pre: phase0.BeaconState) -> List[List[PendingAttestation, MAX_ATTESTATIONS], HISTORICAL_EPOCH_FINALITY_WINDOW * SLOTS_PER_EPOCH]:
    historical_attestations = [[] for _ in range(HISTORICAL_EPOCH_FINALITY_WINDOW * SLOTS_PER_EPOCH)]
    current_slot = pre.slot
    previous_epoch_attestations = pre.previous_epoch_attestations
    current_epoch_attestations  = pre.current_epoch_attestations
    for attestation in current_epoch_attestations:
        attestation_slot = attestation.data.slot
        slots_ago = current_slot - attestation_slot
        if slots_ago < HISTORICAL_EPOCH_FINALITY_WINDOW * SLOTS_PER_EPOCH:
            historical_attestations[slots_ago].append(attestation)
    return historical_attestations    


def populate_historical_chain(pre: phase0.BeaconState) -> List[ChainHistory, HISTORICAL_EPOCH_FINALITY_WINDOW * SLOTS_PER_EPOCH]:
    historical_chain = []
    for i in range(HISTORICAL_EPOCH_FINALITY_WINDOW * SLOTS_PER_EPOCH):
        if pre.slot < i + 1:
            break
        slot_number = pre.slot - i - 1
        curr_root = Bytes32() if slot_number == 0 else get_block_root_at_slot(pre, slot_number)
        parent_root = Bytes32() if slot_number == 0 else get_block_root_at_slot(pre, slot_number - 1)
        parent_slot = 0 if slot_number == 0 else slot_number - 1
        history = ChainHistory(
            block_root=curr_root,
            parent_root=parent_root,
            slot=slot_number,
            parent_slot=parent_slot,
        )
        historical_chain.append(history)
    return historical_chain
    

def upgrade_to_gasper_siesta(pre: phase0.BeaconState) -> BeaconState:
    epoch = phase0.get_current_epoch(pre)
    post = BeaconState(
        # Versioning
        genesis_time=pre.genesis_time,
        genesis_validators_root=pre.genesis_validators_root,
        slot=pre.slot,
        fork=Fork(
            previous_version=pre.fork.current_version,
            current_version=GASPER_SIESTA_FORK_VERSION,
            epoch=epoch,
        ),
        # History
        latest_block_header=pre.latest_block_header,
        block_roots=pre.block_roots,
        state_roots=pre.state_roots,
        historical_roots=pre.historical_roots,
        # Eth1
        eth1_data=pre.eth1_data,
        eth1_data_votes=pre.eth1_data_votes,
        eth1_deposit_index=pre.eth1_deposit_index,
        # Registry
        validators=pre.validators,
        balances=pre.balances,
        # Randomness
        randao_mixes=pre.randao_mixes,
        # Slashings
        slashings=pre.slashings,
        # Attestations
        previous_epoch_attestations=pre.previous_epoch_attestations,
        current_epoch_attestations=pre.current_epoch_attestations,
        # Finality
        justification_bits=pre.justification_bits,
        previous_justified_checkpoint=pre.previous_justified_checkpoint,
        current_justified_checkpoint=pre.current_justified_checkpoint,
        finalized_checkpoint=pre.finalized_checkpoint,
        # Historical epoch attestations and roots
        historical_attestations=populate_historical_attestations(pre),
        historical_chain=populate_historical_chain(pre),
    )
    return post
```