"""WP7 acceptance tests: `ApiLlmAdapter` (plan §3 WP7 row).

Covers: dry-run parsing of canned responses, including malformed ones (the plan's acceptance
criterion); prefix continuation; `Attempt.native` always holding the raw response; the cost cap
(exercised with a locally-defined fake `client` -- never a real network/SDK call, per the plan's "no
network or API calls" rule for this WP); and the declared capability flags.
"""

from __future__ import annotations

import pytest

from code.ndbench.adapters.api_llm import ApiLlmAdapter, CostCapExceeded
from code.ndbench.adapters.base import Capability
from code.ndbench.core.types import PrefixSpec, SampleRequest, Sequent

_SEQ = Sequent(premises=("( P v Q )", "( ~ P )"), conclusion="Q")
_GOOD_RESPONSE = (
    "Sure, here's a proof:\n"
    "N1 ( P v Q ) : PR ; N2 ( ~ P ) : PR ; N3 | P : AS ; N4 | F : NEGE N3 N2 ; "
    "N5 | Q : BOTE N4 ; N6 | Q : AS ; N7 Q : ORE N1 N3 N5 N6 N6 ; QED\n"
    "Hope that helps!"
)


def test_capability_flags() -> None:
    adapter = ApiLlmAdapter(dry_run=True)
    assert adapter.capabilities == (Capability.SAMPLE | Capability.PREFIX)


def test_dry_run_requires_no_client() -> None:
    ApiLlmAdapter(dry_run=True)  # must not raise
    with pytest.raises(ValueError):
        ApiLlmAdapter(dry_run=False)  # no client given


def test_dry_run_parses_a_wellformed_canned_response() -> None:
    adapter = ApiLlmAdapter(dry_run=True, dry_run_responses=[_GOOD_RESPONSE])
    req = SampleRequest(pid="p", sequent=_SEQ, n=1, temperature=0.8, max_new_tokens=200, seed=0)
    (attempts,) = adapter.sample([req])
    a = attempts[0]
    assert a.body == (
        "N1 ( P v Q ) : PR ; N2 ( ~ P ) : PR ; N3 | P : AS ; N4 | F : NEGE N3 N2 ; "
        "N5 | Q : BOTE N4 ; N6 | Q : AS ; N7 Q : ORE N1 N3 N5 N6 N6 ; QED"
    )
    assert a.native == _GOOD_RESPONSE
    assert a.finished is True


@pytest.mark.parametrize(
    "malformed",
    [
        "I don't know how to prove this.",
        "N1 ( P v Q ) : PR ; N2 ( ~ P ) : PR ; ... and then it should follow.",  # never reaches QED
        "The answer is QED, obviously.",  # QED with no preceding N<k> span
        "",
    ],
)
def test_dry_run_malformed_responses_are_unparseable(malformed: str) -> None:
    adapter = ApiLlmAdapter(dry_run=True, dry_run_responses=[malformed])
    req = SampleRequest(pid="p", sequent=_SEQ, n=1, temperature=0.8, max_new_tokens=200, seed=0)
    (attempts,) = adapter.sample([req])
    a = attempts[0]
    assert a.body.startswith("UNPARSEABLE")
    assert a.native == malformed  # the raw response is kept even when parsing fails


def test_dry_run_responses_cycle_through_a_pool() -> None:
    pool = [_GOOD_RESPONSE, "nonsense, no proof here"]
    adapter = ApiLlmAdapter(dry_run=True, dry_run_responses=pool)
    req = SampleRequest(pid="p", sequent=_SEQ, n=4, temperature=0.8, max_new_tokens=200, seed=0)
    (attempts,) = adapter.sample([req])
    assert not attempts[0].body.startswith("UNPARSEABLE")
    assert attempts[1].body.startswith("UNPARSEABLE")
    assert not attempts[2].body.startswith("UNPARSEABLE")
    assert attempts[3].body.startswith("UNPARSEABLE")


def test_dry_run_responses_callable_gets_sequent_and_index() -> None:
    seen: list[tuple[Sequent, int]] = []

    def canned(seq: Sequent, i: int) -> str:
        seen.append((seq, i))
        return _GOOD_RESPONSE

    adapter = ApiLlmAdapter(dry_run=True, dry_run_responses=canned)
    req = SampleRequest(pid="p", sequent=_SEQ, n=3, temperature=0.8, max_new_tokens=200, seed=0)
    adapter.sample([req])
    assert seen == [(_SEQ, 0), (_SEQ, 1), (_SEQ, 2)]


def test_default_dry_run_response_without_dry_run_responses_is_unparseable() -> None:
    adapter = ApiLlmAdapter(dry_run=True)
    req = SampleRequest(pid="p", sequent=_SEQ, n=1, temperature=0.8, max_new_tokens=200, seed=0)
    (attempts,) = adapter.sample([req])
    assert attempts[0].body.startswith("UNPARSEABLE")


def test_prefix_continuation_is_spliced_onto_the_forced_prefix() -> None:
    prefix_body = "N1 ( P v Q ) : PR ; N2 ( ~ P ) : PR ;"
    continuation_response = (
        "N3 | P : AS ; N4 | F : NEGE N3 N2 ; N5 | Q : BOTE N4 ; N6 | Q : AS ; "
        "N7 Q : ORE N1 N3 N5 N6 N6 ; QED"
    )
    adapter = ApiLlmAdapter(dry_run=True, dry_run_responses=[continuation_response])
    req = SampleRequest(
        pid="p",
        sequent=_SEQ,
        n=1,
        temperature=0.8,
        max_new_tokens=200,
        seed=0,
        prefix=PrefixSpec(j=2, body=prefix_body),
    )
    (attempts,) = adapter.sample([req])
    assert attempts[0].body == f"{prefix_body} {continuation_response}"


def test_prefix_prompt_includes_the_forced_prefix_text() -> None:
    adapter = ApiLlmAdapter(dry_run=True, dry_run_responses=["irrelevant QED"])
    prompt = adapter._build_prompt(_SEQ, "N1 ( P v Q ) : PR ;")
    assert "N1 ( P v Q ) : PR ;" in prompt
    assert _SEQ.prompt() in prompt


def test_cost_cap_of_zero_blocks_the_first_real_call() -> None:
    calls = []

    def fake_client(prompt: str) -> str:
        calls.append(prompt)
        return _GOOD_RESPONSE

    adapter = ApiLlmAdapter(client=fake_client, cost_cap_usd=0.0, price_per_1k_tokens=1.0)
    req = SampleRequest(pid="p", sequent=_SEQ, n=1, temperature=0.8, max_new_tokens=200, seed=0)
    with pytest.raises(CostCapExceeded):
        adapter.sample([req])
    assert calls == []  # the cap tripped before any (fake, local -- no network) call was made


def test_cost_cap_trips_after_the_first_call_once_spend_crosses_it() -> None:
    calls = []

    def fake_client(prompt: str) -> str:
        calls.append(prompt)
        return _GOOD_RESPONSE

    adapter = ApiLlmAdapter(client=fake_client, cost_cap_usd=1.0, price_per_1k_tokens=1_000_000.0)
    req = SampleRequest(pid="p", sequent=_SEQ, n=3, temperature=0.8, max_new_tokens=200, seed=0)
    with pytest.raises(CostCapExceeded):
        adapter.sample([req])
    assert len(calls) == 1  # exactly one real call happened before the cap tripped


def test_non_dry_run_uses_injected_client_and_records_native() -> None:
    adapter = ApiLlmAdapter(client=lambda prompt: _GOOD_RESPONSE)
    req = SampleRequest(pid="p", sequent=_SEQ, n=2, temperature=0.8, max_new_tokens=200, seed=0)
    (attempts,) = adapter.sample([req])
    assert len(attempts) == 2
    for a in attempts:
        assert a.native == _GOOD_RESPONSE
        assert not a.body.startswith("UNPARSEABLE")


def test_identity_reports_dry_run_and_cost_cap() -> None:
    adapter = ApiLlmAdapter(dry_run=True, cost_cap_usd=5.0, model_identity={"model": "test-1"})
    ident = adapter.identity()
    assert ident["dry_run"] is True
    assert ident["cost_cap_usd"] == 5.0
    assert ident["model"] == "test-1"
