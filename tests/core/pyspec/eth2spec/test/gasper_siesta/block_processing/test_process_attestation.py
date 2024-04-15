from eth2spec.test.context import (
    spec_state_test,
    always_bls, never_bls,
    with_phases, 
    spec_test,
    low_balances,
    with_custom_state,
    single_phase,
)
from eth2spec.test.helpers.attestations import (
    run_attestation_processing,
    get_valid_attestation,
    get_valid_attestation_at_source_target,
    sign_aggregate_attestation,
    sign_attestation,
    compute_max_inclusion_slot,
)
from eth2spec.test.helpers.constants import GASPER_SIESTA
from eth2spec.test.helpers.state import (
    next_slots,
    next_epoch_via_block,
    transition_to_slot_via_block,
)
from eth2spec.utils.ssz.ssz_impl import uint_to_bytes
from eth2spec.utils.ssz.ssz_typing import Bitlist, Bytes32
from remerkleable.basic import uint256

@with_phases(phases=[GASPER_SIESTA])
@spec_state_test
def test_one_basic_attestation(spec, state):
    attestation = get_valid_attestation(spec, state, signed=True)
    print("Attestation is : ", attestation)
    next_slots(spec, state, spec.MIN_ATTESTATION_INCLUSION_DELAY)
    print("State after slot is: ", state.slot)
    print("State historical epoch attestations is: ", state.historical_epoch_attestations)
    print("State current block roots is: ", state.block_roots)

    yield from run_attestation_processing(spec, state, attestation)

@with_phases(phases=[GASPER_SIESTA])
@spec_state_test
def test_one_basic_attestation_source_target(spec, state):
    source_root = Bytes32(uint_to_bytes(uint256(0)))
    target_root = Bytes32(uint_to_bytes(uint256(1)))
    attestation = get_valid_attestation_at_source_target(spec, state, source_epoch=0, target_epoch=0, source_root=source_root, target_root=target_root, signed=True)
    print("Attestation is : ", attestation)
    next_slots(spec, state, spec.MIN_ATTESTATION_INCLUSION_DELAY)
    
    yield from run_attestation_processing(spec, state, attestation)

    print("State after slot is: ", state.slot)
    print("State historical epoch attestations is: ", state.historical_epoch_attestations)
    print("State current block roots is: ", state.block_roots)

