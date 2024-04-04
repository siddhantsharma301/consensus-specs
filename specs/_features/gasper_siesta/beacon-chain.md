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
  - [[CATEGORY OF HELPERS]](#category-of-helpers)
  - [Epoch processing](#epoch-processing)
  - [Block processing](#block-processing)
- [Testing](#testing)

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

<!-- 
## Configuration

### [CATEGORY OF CONFIGURATIONS]

| Name | Value |
| - | - |
| `<CONFIGURATION_FIELD_NAME>` | `<VALUE>` | 
-->

## Containers

### Beacon state

#### `BeaconState`

```python
class BeaconState(phase0.BeaconState):
    historical_epoch_attestations: Vector[Attestation, MAX_ATTESTATIONS * HISTORICAL_EPOCH_FINALITY_WINDOW]
    historical_epoch_block_roots: Vector[Root, HISTORICAL_EPOCH_FINALITY_WINDOW]
```

## Helper functions

### [CATEGORY OF HELPERS]

```python
<PYTHON HELPER FUNCTION>
```

### Epoch processing


### Block processing

    

    
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
