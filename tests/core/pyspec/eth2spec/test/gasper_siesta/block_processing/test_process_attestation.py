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
from eth2spec.utils.ssz.ssz_typing import Bitlist

@with_phases(phases=[GASPER_SIESTA])
@spec_state_test
def test_one_basic_attestation(spec, state):
    attestation = get_valid_attestation(spec, state, signed=True)
    print("Attestation is : ", attestation)
    next_slots(spec, state, spec.MIN_ATTESTATION_INCLUSION_DELAY)
    print("State after slot is: ", state.slot)

    yield from run_attestation_processing(spec, state, attestation)
