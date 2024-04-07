# Gasper-Siesta -- Fork Logic

## Table of contents

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Gasper-Siesta -- Fork Logic](#gasper-siesta----fork-logic)
  - [Table of contents](#table-of-contents)
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
def create_dummy_pending_attestation() -> PendingAttestation:
    return PendingAttestation(
        aggregation_bits=Bitlist[MAX_VALIDATORS_PER_COMMITTEE](),
        data=AttestationData(
            slot=Slot(0),
            index=CommitteeIndex(0),
            beacon_block_root=Root(),
            source=Checkpoint(epoch=Epoch(0), root=Root()),
            target=Checkpoint(epoch=Epoch(0), root=Root()),
        ),
        inclusion_delay=Slot(0),
        proposer_index=ValidatorIndex(0),
    )

def populate_historical_epoch_attestations(pre: phase0.BeaconState) -> Vector[Vector[PendingAttestation, MAX_ATTESTATIONS], HISTORICAL_EPOCH_FINALITY_WINDOW]:
    """
    Populate the historical_epoch_attestations with attestations from the end of every epoch
    in the pre-state container. For epochs beyond the last two, where specific attestations
    cannot be retrieved, placeholders will be used.
    """
    historical_epoch_attestations = []
    current_epoch = phase0.get_current_epoch(pre)
    previous_epoch = phase0.get_previous_epoch(pre)

    # Populate for the last two epochs with actual data
    if current_epoch > 0:
        current_epoch_attestations = get_matching_target_attestations(pre, current_epoch)
        assert len(current_epoch_attestations) <= MAX_ATTESTATIONS, "Exceeded MAX_ATTESTATIONS for current epoch"
        historical_epoch_attestations.append([current_epoch_attestations])
    if previous_epoch > 0 and previous_epoch != current_epoch:
        previous_epoch_attestations = get_matching_target_attestations(pre, previous_epoch)
        assert len(previous_epoch_attestations) <= MAX_ATTESTATIONS, "Exceeded MAX_ATTESTATIONS for previous epoch"
        historical_epoch_attestations.append([previous_epoch_attestations])  # Insert at the beginning

    # Placeholder for other epochs
    placeholder_attestations = [create_dummy_pending_attestation()]
    while len(historical_epoch_attestations) < HISTORICAL_EPOCH_FINALITY_WINDOW:
        historical_epoch_attestations.append(placeholder_attestations)  # Insert at the beginning to maintain chronological order

    return historical_epoch_attestations

def get_block_root_at_epoch(pre: BeaconState, epoch: Epoch) -> Root:
    # Calculate the slot at the end of the epoch
    epoch_end_slot = (epoch + 1) * SLOTS_PER_EPOCH - 1
    # Ensure the requested epoch is within the historical bounds
    assert epoch <= state.current_epoch() + HISTORICAL_ROOTS_LIMIT // SLOTS_PER_EPOCH, "Requested epoch is too far in history"
    # Index into the state's block roots array to retrieve the block root
    block_root_index = epoch_end_slot % SLOTS_PER_HISTORICAL_ROOT
    block_root = state.block_roots[block_root_index]
    return block_root

def populate_historical_epoch_block_roots(pre: phase0.BeaconState) -> Vector[Root, HISTORICAL_EPOCH_FINALITY_WINDOW]:
    """
    Populate the historical_epoch_block_roots with block roots from the end of every epoch
    in the pre-state container.
    """
    historical_epoch_block_roots = []
    current_epoch = phase0.get_current_epoch(pre)
    start_epoch = max(0, current_epoch - HISTORICAL_EPOCH_FINALITY_WINDOW)
    for epoch in range(start_epoch, current_epoch):
        epoch_block_root = get_block_root_at_epoch(pre, epoch)
        historical_epoch_block_roots.append(epoch_block_root)
        # Ensure the list is capped at the size defined by HISTORICAL_EPOCH_FINALITY_WINDOW
        # This is necessary if the number of collected block roots exceeds the storage limit.
        historical_epoch_block_roots = historical_epoch_block_roots[:HISTORICAL_EPOCH_FINALITY_WINDOW]
    return historical_epoch_block_roots

# TODO: FIX THIS FUNCTION
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
        historical_epoch_attestations=populate_historical_epoch_attestations(pre),
        historical_epoch_block_roots=populate_historical_epoch_block_roots(pre),
    )
    return post
```
