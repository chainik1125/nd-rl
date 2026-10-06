"""Load-bearing isolation, exact retrieval, and hard-limit protocol tests."""

import pytest

from code.shared_code.learned_allocation import Attempt, Decision, Episode, PublicAttempt


def attempt(text: str, proof: str = "invalid", size: int = 3) -> Attempt:
    return Attempt(PublicAttempt((7,) * size, text), proof, False, True, (-0.2,) * size)


def test_visible_history_has_no_private_proof_or_verifier_metadata() -> None:
    episode = Episode("original theorem")
    episode.append(attempt("raw output", "private converted proof"))
    theorem, history = episode.observe()
    assert theorem == "original theorem"
    assert history == (PublicAttempt((7, 7, 7), "raw output"),)
    assert not hasattr(history[0], "proof")
    assert not hasattr(history[0], "complete")


def test_correct_attempt_does_not_automatically_stop_and_original_theorem_is_used() -> None:
    episode = Episode("original theorem", attempt_cost=0.1)
    episode.append(attempt("first", "good"))
    assert episode.decide(Decision(None, -0.7, 0.5), lambda *_: True, 3) is None
    episode.append(attempt("second", "new premises"))
    calls = []

    def verify(theorem: str, proof: str) -> bool:
        calls.append((theorem, proof))
        return proof == "good"

    result = episode.decide(Decision(0, -0.8, 0.8), verify, 3)
    assert calls == [("original theorem", "good")]
    assert result is not None
    assert result.correct and result.reward == pytest.approx(0.9)
    assert not result.forced
    assert result.attempts == 2


def test_limit_masks_continue_and_forced_selection_has_same_reward() -> None:
    episode = Episode("theorem", max_attempts=2, attempt_cost=0.03)
    episode.append(attempt("one"))
    episode.decide(Decision(None, -1, 0.5), lambda *_: False, 3)
    episode.append(attempt("two"))
    with pytest.raises(ValueError, match="masked"):
        episode.decide(Decision(None, -1, 0.5), lambda *_: False, 3)
    result = episode.decide(Decision(1, -0.5, 1), lambda *_: False, 3, forced=True)
    assert result is not None
    assert result.reward == -0.03 and result.forced
    assert result.termination_reason == "attempt_limit"


def test_context_space_is_reserved_without_dropping_history() -> None:
    episode = Episode("theorem", context_limit=12, selection_reserve=4, initial_tokens=2)
    episode.append(attempt("one", size=4))
    assert not episode.can_continue(3)
    assert episode.can_continue(2)
    with pytest.raises(ValueError, match="reserved"):
        other = Episode("theorem", context_limit=12, selection_reserve=4, initial_tokens=2)
        other.append(attempt("too long", size=7))
    assert len(episode.observe()[1]) == 1
    result = episode.decide(Decision(0, 0, 1), lambda *_: True, 3, forced=True)
    assert result is not None and result.termination_reason == "context_limit"


def test_invalid_likelihood_and_index_rejected() -> None:
    with pytest.raises(ValueError, match="likelihood"):
        Attempt(PublicAttempt((1,), "raw"), "proof", True, False, (-1, -2))
    episode = Episode("theorem")
    episode.append(attempt("one"))
    with pytest.raises(ValueError, match="index"):
        episode.decide(Decision(1, -1, 0.5), lambda *_: True, 3)


def test_hard_limit_is_forced_even_when_caller_omits_flag() -> None:
    episode = Episode("theorem", max_attempts=1)
    episode.append(attempt("one"))
    result = episode.decide(Decision(0, 0, 1), lambda *_: True, 3)
    assert result is not None and result.forced


def test_attempt_and_decision_alternate() -> None:
    episode = Episode("theorem")
    episode.append(attempt("one"))
    with pytest.raises(ValueError):
        episode.append(attempt("two"))
    episode.decide(Decision(None, -1, 0.5), lambda *_: False, 3)
    with pytest.raises(ValueError):
        episode.decide(Decision(None, -1, 0.5), lambda *_: False, 3)
