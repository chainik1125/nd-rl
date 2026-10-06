"""Masked behavior likelihoods, including forced-stop gradient isolation."""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch.distributions import Bernoulli, Categorical

_CONTROLLER = (
    Path(__file__).parents[1]
    / "experiments/current/learned_compute_allocation_20261006/allocation/controller.py"
)
_SPEC = importlib.util.spec_from_file_location("allocation_controller", _CONTROLLER)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_forced_accept_has_selection_likelihood_only_and_no_stopping_gradient() -> None:
    stop_logit = torch.tensor([0.7], requires_grad=True)
    select_logits = torch.tensor([[0.2, -0.3]], requires_grad=True)
    probability = _MODULE.Controller.action_logprob(
        Bernoulli(logits=stop_logit),
        Categorical(logits=select_logits),
        torch.ones(1),
        torch.tensor([1]),
        torch.tensor([True]),
    )
    expected = torch.log_softmax(select_logits, -1)[0, 1]
    assert torch.allclose(probability[0], expected)
    probability.sum().backward()
    assert stop_logit.grad is not None and stop_logit.grad.item() == 0
    assert select_logits.grad is not None and select_logits.grad.abs().sum() > 0


def test_continue_has_stopping_likelihood_only() -> None:
    stop_logit = torch.tensor([0.7], requires_grad=True)
    select_logits = torch.tensor([[0.2, -0.3]], requires_grad=True)
    probability = _MODULE.Controller.action_logprob(
        Bernoulli(logits=stop_logit),
        Categorical(logits=select_logits),
        torch.zeros(1),
        torch.tensor([0]),
        torch.tensor([False]),
    )
    assert torch.allclose(probability, torch.nn.functional.logsigmoid(-stop_logit))
    probability.sum().backward()
    assert select_logits.grad is not None and select_logits.grad.abs().sum() == 0


def test_impossible_forced_continue_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="impossible"):
        _MODULE.Controller.action_logprob(
            Bernoulli(logits=torch.tensor([0.0])),
            Categorical(logits=torch.tensor([[0.0]])),
            torch.zeros(1),
            torch.zeros(1, dtype=torch.long),
            torch.tensor([True]),
        )


def test_generation_replay_scores_all_raw_tokens_and_temperature() -> None:
    live_spec = importlib.util.spec_from_file_location(
        "allocation_live", _CONTROLLER.with_name("live.py")
    )
    assert live_spec is not None and live_spec.loader is not None
    live = importlib.util.module_from_spec(live_spec)
    live_spec.loader.exec_module(live)

    class ToyBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([0.5, -0.1, 1.2]))

        def forward(self, ids):
            return self.logits[None, None].expand(ids.shape[0], ids.shape[1], -1)

    base = ToyBase()
    # Zero is a sampled PAD and one an EOS; both must contribute likelihood.
    raw = [2, 0, 1]
    lp, entropy = live.token_logprobs(base, [1, 2], raw, "cpu", temperature=0.8)
    expected = torch.log_softmax(base.logits / 0.8, -1)[torch.tensor(raw)]
    assert torch.allclose(lp, expected) and len(lp) == len(raw)
    _, unit_entropy = live.token_logprobs(base, [1, 2], raw, "cpu", temperature=1)
    assert torch.allclose(entropy, unit_entropy)
    (-lp.sum()).backward()
    assert base.logits.grad is not None and base.logits.grad.abs().sum() > 0


@pytest.mark.parametrize("truncated", [False, True])
def test_live_compute_counts_execution_and_separates_diagnostic_replay(
    monkeypatch, truncated
) -> None:
    import random
    import sys
    from types import SimpleNamespace

    live_spec = importlib.util.spec_from_file_location(
        "allocation_live_counts", _CONTROLLER.with_name("live.py")
    )
    assert live_spec is not None and live_spec.loader is not None
    live = importlib.util.module_from_spec(live_spec)
    live_spec.loader.exec_module(live)

    class ToyBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([0.0, 1.0, 0.0]))
            self.cfg = {"n_head": 8}

        def forward(self, ids, **kwargs):
            return self.logits[None, None].expand(ids.shape[0], ids.shape[1], -1)

    class ToyEnv:
        def __init__(self, *args, **kwargs):
            self.done = False
            self._eff = []
            self.text = []
            self.base = 0

        def apply(self, action):
            # A failing transition can still canonicalize and append public text.
            self._eff = ["P"]
            self.text = ["P"]
            return False, "private error"

    class ToyTok:
        eos = 1
        pad = 0
        itos = ["<pad>", "<eos>", "P"]

        def decode_action(self, raw):
            return ["<pad>"], not truncated

        def encode_toks(self, tokens):
            return [self.itos.index(token) for token in tokens]

        def text(self, tokens):
            return " ".join(tokens)

    def sampler(base, tok, prompts, **kwargs):
        base(torch.tensor([prompts[0]]), caches=[{"n": 0}])
        # This extra executed padding decode is absent from the visible raw proof.
        base(torch.tensor([[tok.pad]]), caches=[{"n": len(prompts[0])}])
        return [[tok.pad, 2]] if truncated else [[tok.pad, tok.eos, tok.pad]]

    monkeypatch.setitem(sys.modules, "sample", SimpleNamespace(generate_ids_fast=sampler))
    monkeypatch.setitem(sys.modules, "state_env", SimpleNamespace(Env=ToyEnv))
    monkeypatch.setitem(sys.modules, "state_sample", SimpleNamespace(prompt_ids=lambda *_: [2]))
    candidate, private, actions = live.generate_attempt(
        SimpleNamespace(base=ToyBase()), ToyTok(), "original theorem", [], random.Random(11)
    )
    assert candidate["generated_tokens"] == 2
    assert candidate["forward_calls"] == 2
    assert candidate["forward_tokens"] == 2
    assert candidate["attention_score_cells_per_layer"] == 24
    assert candidate["likelihood_replay_forward_calls"] == 1
    assert candidate["likelihood_replay_forward_tokens"] == 2
    assert len(actions) == 1
    if truncated:
        assert private["termination"] == "action_truncated"
        assert candidate["transition_tokens"] == [0, 2]
        assert candidate["effective_tokens"] == [0, 2]
    else:
        assert private["termination"] == "grammar_failure"
        assert candidate["transition_tokens"] == [2, 2, 1]
        assert candidate["effective_tokens"] == [2, 1]
        assert candidate["raw_tokens"] == [0, 1]


def test_full_path_integration_matches_enumerated_policy_and_compute(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(_CONTROLLER.parent))
    spec = importlib.util.spec_from_file_location(
        "allocation_path_integration", _CONTROLLER.with_name("evaluate_live_path.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = {
        "decisions": [
            {
                "forced": False,
                "accept_probability": 0.3,
                "selection_probabilities": [1.0],
                "ids": [1],
                "controller_wall_seconds": 0,
            },
            {
                "forced": True,
                "accept_probability": 0.1,
                "selection_probabilities": [0.4, 0.6],
                "ids": [1, 2],
                "controller_wall_seconds": 0,
            },
        ],
        "candidates": [
            dict.fromkeys(
                (
                    "generated_tokens",
                    "forward_calls",
                    "forward_tokens",
                    "attention_score_cells_per_layer",
                    "sampling_wall_seconds",
                ),
                work,
            )
            for work in (10, 20)
        ],
    }
    result = module.integrate(path, [False, True], cost=0.1)
    assert result["correct_probability"] == pytest.approx(0.7 * 0.6)
    assert result["attempts"] == pytest.approx(1.7)
    assert result["reward"] == pytest.approx(0.42 - 0.1 * 0.7)
    assert result["forced_termination_probability"] == pytest.approx(0.7)
    assert result["expected_policy_work"]["forward_tokens"] == pytest.approx(24)
    assert result["expected_policy_work"]["controller_forward_tokens"] == pytest.approx(2.4)
    fixed = module.integrate(path, [False, True], method="fixed2")
    assert fixed["correct_probability"] == pytest.approx(0.6)
    assert fixed["attempts"] == 2
    assert fixed["expected_policy_work"]["controller_forward_calls"] == 1
    assert fixed["expected_policy_work"]["controller_forward_tokens"] == 2
    counter = module.integrate(path, [False, True], method="counter", hazards=[0.8, 0.1])
    assert counter["correct_probability"] == pytest.approx(0.2 * 0.6)
    assert counter["attempts"] == pytest.approx(1.2)
    assert counter["expected_policy_work"]["controller_forward_calls"] == pytest.approx(0.2)
    assert counter["expected_policy_work"]["controller_forward_tokens"] == pytest.approx(0.4)
    fixed1 = module.integrate(path, [False, True], method="fixed1")
    assert fixed1["expected_policy_work"]["controller_forward_calls"] == 0
    short_context = module.integrate(path, [False, True], method="context1")
    assert short_context["attempts"] == 1
    assert short_context["expected_policy_work"]["controller_forward_calls"] == 0
    long_context = module.integrate(path, [False, True], method="context3")
    assert long_context["attempts"] == 2
    assert long_context["correct_probability"] == pytest.approx(0.6)
    greedy = module.integrate(path, [False, True], method="learned_greedy_index")
    assert greedy["correct_probability"] == pytest.approx(0.7)
    fully_greedy = module.integrate(path, [False, True], method="greedy_stop_greedy_index")
    assert fully_greedy["correct_probability"] == 1
    assert fully_greedy["attempts"] == 2
    path["decisions"][1]["selection_probabilities"] = [0.5, 0.5]
    tied = module.integrate(path, [False, True], method="fixed2_greedy_index")
    assert tied["correct_probability"] == 0.5


def test_test_freeze_and_counter_provenance_bind_model_protocol_and_dataset() -> None:
    spec = importlib.util.spec_from_file_location(
        "allocation_eval_guard", _CONTROLLER.with_name("evaluation_protocol.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = dict.fromkeys(module.INFERENCE_KEYS, 1)
    config.update(
        base_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
        runner_hashes=dict.fromkeys(module.BEHAVIOR_FILES, "h" * 64),
        source_hashes={"state_env.py": "s" * 64},
    )
    value = module.signature(config)
    digest = module.fingerprint(value)
    freeze = {
        "status": "frozen",
        "selection_split": "dev",
        "test_data_sha256": "t" * 64,
        "test_seed": 62006,
        "test_repeats": 2,
        "test_theorems": 400,
        "registered_models": [
            {"base_sha256": "b" * 64, "checkpoint_sha256": "c" * 64, "inference_sha256": digest}
        ],
    }
    module.validate_test_freeze(freeze, value, "t" * 64, 62006, 2, 400)
    with pytest.raises(ValueError, match="theorem count"):
        module.validate_test_freeze(freeze, value, "t" * 64, 62006, 2, 1)
    for seed, repeats in ((52006, 2), (62006, 1)):
        with pytest.raises(ValueError, match="sampling draws"):
            module.validate_test_freeze(freeze, value, "t" * 64, seed, repeats, 400)
    for name in module.BEHAVIOR_FILES:
        changed = dict(config)
        changed["runner_hashes"] = config["runner_hashes"] | {name: "x" * 64}
        with pytest.raises(ValueError, match="not registered"):
            module.validate_test_freeze(freeze, module.signature(changed), "t" * 64, 62006, 2, 400)
    with pytest.raises(ValueError, match="dataset"):
        module.validate_test_freeze(freeze, value, "x" * 64, 62006, 2, 400)
    with pytest.raises(ValueError, match="not registered"):
        module.validate_test_freeze(
            freeze, value | {"checkpoint_sha256": "x" * 64}, "t" * 64, 62006, 2, 400
        )
    counter = {
        "fit_split": "dev",
        "base_sha256": "b" * 64,
        "checkpoint_sha256": "c" * 64,
        "inference_sha256": digest,
        "fit_data_sha256": "d" * 64,
        "fit_public_paths_sha256": "p" * 64,
    }
    module.validate_counter(counter, value)
    with pytest.raises(ValueError, match="different model"):
        module.validate_counter(counter, value | {"checkpoint_sha256": "x" * 64})
    with pytest.raises(ValueError, match="different inference"):
        module.validate_counter(counter, value | {"inference": {"temperature": 0.9}})


def test_fixed_mixtures_match_dev_budgets_without_inventing_feasible_matches(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(_CONTROLLER.parent))
    spec = importlib.util.spec_from_file_location(
        "allocation_budget_fit", _CONTROLLER.with_name("fit_budget_baselines.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixed = {
        str(k): {"attempts": float(k), "total_forward_tokens": float(k * k), "reward": k / 10}
        for k in (1, 2, 4, 8)
    }
    for metric, target in (("attempts", 3.0), ("total_forward_tokens", 10.0)):
        fit = module.matched_mixture(fixed, target, metric)
        assert fit["feasible"]
        assert sum(fit["weights"].values()) == pytest.approx(1)
        assert sum(fit["weights"][k] * row[metric] for k, row in fixed.items()) == pytest.approx(
            target
        )
    joint = module.joint_mixture(fixed, {"attempts": 3.0, "total_forward_tokens": 11.0})
    assert joint["feasible"]
    for metric, target in (("attempts", 3.0), ("total_forward_tokens", 11.0)):
        assert sum(joint["weights"][k] * row[metric] for k, row in fixed.items()) == pytest.approx(
            target
        )
    assert not module.matched_mixture(fixed, 9.0, "attempts")["feasible"]
    assert not module.joint_mixture(fixed, {"attempts": 3.0, "total_forward_tokens": 1.0})[
        "feasible"
    ]


def test_saved_path_and_postprocessing_guards_bind_actual_implementations() -> None:
    import hashlib

    spec = importlib.util.spec_from_file_location(
        "allocation_post_guard", _CONTROLLER.with_name("evaluation_protocol.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = dict.fromkeys(module.INFERENCE_KEYS, 1)
    config.update(
        base_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
        data_sha256="t" * 64,
        source_hashes={},
        runner_hashes={
            name: hashlib.sha256(_CONTROLLER.with_name(name).read_bytes()).hexdigest()
            for name in module.BEHAVIOR_FILES
        },
    )
    module.validate_saved_implementation(config)
    stale = config | {
        "runner_hashes": config["runner_hashes"] | {"evaluate_live_path.py": "x" * 64}
    }
    with pytest.raises(ValueError, match="different behavior"):
        module.validate_saved_implementation(stale)
    freeze = {
        "status": "frozen",
        "selection_split": "dev",
        "test_data_sha256": "t" * 64,
        "registered_models": [
            {
                "base_sha256": "b" * 64,
                "checkpoint_sha256": "c" * 64,
                "inference_sha256": module.fingerprint(module.signature(config)),
                "budget_baselines_sha256": "a" * 64,
            }
        ],
        "postprocessing_hashes": {"analysis.py": "p" * 64},
    }
    module.validate_postprocessing_registration(
        freeze, config, "a" * 64, "budget_baselines_sha256", {"analysis.py": "p" * 64}
    )
    with pytest.raises(ValueError, match="baseline is not registered"):
        module.validate_postprocessing_registration(
            freeze, config, "x" * 64, "budget_baselines_sha256", {"analysis.py": "p" * 64}
        )
    with pytest.raises(ValueError, match="implementation is not registered"):
        module.validate_postprocessing_registration(
            freeze, config, "a" * 64, "budget_baselines_sha256", {"analysis.py": "x" * 64}
        )


def test_statistics_average_draws_then_seeds_and_bootstrap_whole_family_strata(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(_CONTROLLER.parent))
    spec = importlib.util.spec_from_file_location(
        "allocation_statistics", _CONTROLLER.with_name("aggregate_live.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import numpy as np

    work = dict.fromkeys(
        (
            "generated_tokens",
            "forward_tokens",
            "controller_forward_tokens",
            "forward_calls",
            "controller_forward_calls",
            "attention_score_cells_per_layer",
            "controller_attention_score_cells_per_layer",
            "policy_wall_seconds",
        ),
        0.0,
    )
    rows = []
    for theorem_id, probabilities in (("a", (0.0, 1.0)), ("b", (1.0, 1.0))):
        for repeat, probability in enumerate(probabilities):
            rows.append(
                {
                    "theorem_id": theorem_id,
                    "repeat": repeat,
                    "correct_probability": probability,
                    "attempts": 1.0,
                    "reward": probability,
                    "forced_termination_probability": 0.0,
                    "expected_policy_work": work,
                }
            )
    means = module.draw_means(rows, ["a", "b"], 2)
    assert means[:, 0].tolist() == [0.5, 1.0]
    across_seeds = module.arm_mean([{"learned": means}, {"learned": means * 0.5}])
    assert across_seeds["learned"][:, 0].tolist() == [0.375, 0.75]
    with pytest.raises(ValueError, match="duplicate"):
        module.draw_means(rows + [rows[0]], ["a", "b"], 2)
    with pytest.raises(ValueError, match="missing"):
        module.draw_means(rows[:-1], ["a", "b"], 2)
    weights = module.bootstrap_weights(["f", "f", "o", "o"], samples=512, seed=9)
    assert np.allclose(weights[:, :2].sum(axis=1), 0.5)
    assert np.allclose(weights[:, 2:].sum(axis=1), 0.5)
    interval = module.estimate(np.array([[0.1], [0.1], [0.3], [0.3]]), weights, names=("delta",))
    assert interval["delta"]["mean"] == pytest.approx(0.2)
    assert interval["delta"]["ci_low"] == pytest.approx(0.2)
    assert interval["delta"]["ci_high"] == pytest.approx(0.2)
    availability = module.availability_means(
        [
            {"theorem_id": "a", "repeat": 0, "validities": [False, True]},
            {"theorem_id": "a", "repeat": 1, "validities": [True, False]},
        ],
        ["a"],
        2,
    )
    assert availability.tolist() == [[0.5, 1.0, 1.0, 1.0]]


def test_statistics_require_registered_code_settings_and_family_file(monkeypatch, tmp_path) -> None:
    monkeypatch.syspath_prepend(str(_CONTROLLER.parent))
    spec = importlib.util.spec_from_file_location(
        "allocation_statistics_guard", _CONTROLLER.with_name("aggregate_live.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    family_file = tmp_path / "families.jsonl"
    family_file.write_text('{"theorem_id":"a","family":"forward"}\n')
    settings = {
        "script_sha256": module.sha(module.__file__),
        "bootstrap_samples": module.BOOTSTRAP_SAMPLES,
        "bootstrap_seed": module.BOOTSTRAP_SEED,
        "confidence": module.CONFIDENCE,
        "family_data_sha256": module.sha(family_file),
    }
    freeze = {"statistical_analysis": settings, "test_seed": 62006, "test_repeats": 2}
    module.validate_statistics(freeze, family_file)
    for key in settings:
        changed = freeze | {"statistical_analysis": settings | {key: "changed"}}
        with pytest.raises(ValueError, match="not frozen"):
            module.validate_statistics(changed, family_file)


def test_registered_format_control_loads_only_after_model_arm_and_seed_guard(
    monkeypatch, tmp_path
) -> None:
    import json

    monkeypatch.syspath_prepend(str(_CONTROLLER.parent))
    spec = importlib.util.spec_from_file_location(
        "allocation_statistics_load", _CONTROLLER.with_name("aggregate_live.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = __import__("evaluation_protocol")
    config = dict.fromkeys(protocol.INFERENCE_KEYS, 1)
    config.update(
        base_sha256="b" * 64,
        checkpoint_sha256=None,
        data_sha256="t" * 64,
        seed=62006,
        repeats=2,
        evaluated_theorems=1,
        evaluation_split="test",
        frozen_protocol_sha256="f" * 64,
        cost=0.0,
        source_hashes={},
        runner_hashes={
            name: module.sha(_CONTROLLER.with_name(name)) for name in protocol.BEHAVIOR_FILES
        },
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    entry = {
        "base_sha256": "b" * 64,
        "checkpoint_sha256": None,
        "inference_sha256": protocol.fingerprint(protocol.signature(config)),
        "arm": "format_only",
        "rl_seed": None,
    }
    freeze = {
        "status": "frozen",
        "selection_split": "dev",
        "test_data_sha256": "t" * 64,
        "test_seed": 62006,
        "test_repeats": 2,
        "test_theorems": 1,
        "registered_models": [entry],
    }
    with pytest.raises(ValueError, match="arm or training seed"):
        module.load_run(
            {"paths": str(tmp_path), "arm": "format_only", "seed": 101}, freeze, "f" * 64, ["a"]
        )
    # Private files did not exist when the incorrect registration was rejected.
    work = dict.fromkeys(
        (
            "generated_tokens",
            "forward_tokens",
            "controller_forward_tokens",
            "forward_calls",
            "controller_forward_calls",
            "attention_score_cells_per_layer",
            "controller_attention_score_cells_per_layer",
            "policy_wall_seconds",
        ),
        0.0,
    )
    rows = [
        {
            "theorem_id": "a",
            "repeat": repeat,
            "correct_probability": float(valid),
            "attempts": 1.0,
            "reward": float(valid),
            "forced_termination_probability": 1.0,
            "expected_policy_work": work,
        }
        for repeat, valid in enumerate((True, False))
    ]
    (tmp_path / "integrated.json").write_text(json.dumps({"fixed1": rows}))
    (tmp_path / "private_labels.jsonl").write_text(
        "\n".join(
            json.dumps({"theorem_id": "a", "repeat": repeat, "validities": [valid]})
            for repeat, valid in enumerate((True, False))
        )
    )
    tables, availability, _ = module.load_run(
        {"paths": str(tmp_path), "arm": "format_only", "seed": None}, freeze, "f" * 64, ["a"]
    )
    assert tables["fixed1"][0, 0] == 0.5
    assert availability.tolist() == [[0.5, 0.5, 0.5, 0.5]]


def test_statistics_reject_invalid_work_probability_budget_and_native_reward(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(_CONTROLLER.parent))
    spec = importlib.util.spec_from_file_location(
        "allocation_statistics_metrics", _CONTROLLER.with_name("aggregate_live.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    work = dict.fromkeys(
        (
            "generated_tokens",
            "forward_tokens",
            "controller_forward_tokens",
            "forward_calls",
            "controller_forward_calls",
            "attention_score_cells_per_layer",
            "controller_attention_score_cells_per_layer",
            "policy_wall_seconds",
        ),
        0.0,
    )
    row = {
        "correct_probability": 0.5,
        "attempts": 2.0,
        "reward": 0.47,
        "forced_termination_probability": 0.1,
        "expected_policy_work": work,
    }
    module.metrics(row, cost=0.03, max_attempts=8)
    invalid = [
        row | {"reward": 0.5},
        row | {"attempts": 9},
        row | {"forced_termination_probability": 1.1},
        row | {"expected_policy_work": work | {"forward_calls": -1}},
    ]
    for value in invalid:
        with pytest.raises(ValueError, match="invalid integrated"):
            module.metrics(value, cost=0.03, max_attempts=8)


def test_no_cost_queue_preflight_rejects_output_symlink_escape(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(_CONTROLLER.parent))
    spec = importlib.util.spec_from_file_location(
        "allocation_dev_queue", _CONTROLLER.with_name("run_no_cost_dev.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ART = tmp_path / "artifacts"
    allowed = module.ART / "allocation/no_cost"
    allowed.mkdir(parents=True)
    plan = module.preflight()
    assert plan["runs"][103]["selection"].is_relative_to(allowed)
    outside = tmp_path / "unrelated_cost_artifact"
    outside.mkdir()
    plan["runs"][103]["outputs"][100].symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        module.preflight()


def test_historical_registry_preserves_main_choices_and_separate_draws(
    tmp_path, monkeypatch
) -> None:
    import json

    monkeypatch.syspath_prepend(str(_CONTROLLER.parent))
    spec = importlib.util.spec_from_file_location(
        "allocation_historical", _CONTROLLER.with_name("aggregate_historical.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entries = [
        {"arm": arm, "rl_seed": seed, "checkpoint_sha256": f"{arm}:{seed}"}
        for arm in ("cost", "no_cost")
        for seed in (101, 102, 103)
    ] + [
        {"arm": "format_only", "rl_seed": None, "checkpoint_sha256": None},
        {"arm": "fixed1_control", "rl_seed": 101, "checkpoint_sha256": "fixed1"},
        {"arm": "fixed4_control", "rl_seed": 101, "checkpoint_sha256": "fixed4"},
    ]
    main = tmp_path / "main.json"
    main.write_text(
        json.dumps({"status": "frozen", "selection_split": "dev", "registered_models": entries})
    )
    roles, exposure = tmp_path / "roles.jsonl", tmp_path / "exposure.json"
    roles.write_text("registered public role metadata")
    exposure.write_text("registered exposure audit")
    freeze = {
        "test_seed": 92006,
        "test_repeats": 2,
        "test_theorems": 72,
        "main_frozen_protocol_sha256": module.primary.sha(main),
        "historical_exposure_audit_sha256": module.primary.sha(exposure),
        "registered_models": entries,
        "statistical_analysis": {
            "script_sha256": module.primary.sha(module.__file__),
            "aggregate_live_dependency_sha256": module.primary.sha(module.primary.__file__),
            "bootstrap_samples": 10000,
            "bootstrap_seed": 102006,
            "confidence": 0.95,
            "family_data_sha256": module.primary.sha(roles),
        },
    }
    module.validate_registration(freeze, main, roles, exposure)
    with pytest.raises(ValueError, match="draw protocol"):
        module.validate_registration(freeze | {"test_seed": 62006}, main, roles, exposure)
    changed = [entry.copy() for entry in entries]
    changed[0]["counter_hazards_sha256"] = "a newly fitted historical policy"
    with pytest.raises(ValueError, match="differ from MAIN"):
        module.validate_registration(freeze | {"registered_models": changed}, main, roles, exposure)


def test_test_launcher_refuses_unsealed_registry_before_inputs(tmp_path, monkeypatch) -> None:
    import json
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(_CONTROLLER.parent))
    spec = importlib.util.spec_from_file_location(
        "allocation_test_launcher", _CONTROLLER.with_name("launch_live_test.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry = tmp_path / "draft.json"
    registry.write_text(json.dumps({"status": "draft_pending", "selection_split": "dev"}))
    nonexistent_input = tmp_path / "unopened_test_input.jsonl"
    with pytest.raises(ValueError, match="seal the registry"):
        module.preflight(SimpleNamespace(registry=str(registry), data=str(nonexistent_input)))
    assert not nonexistent_input.exists()
    assert list(tmp_path.iterdir()) == [registry]


def test_test_launcher_refuses_occupied_or_unassigned_gpu(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(_CONTROLLER.parent))
    spec = importlib.util.spec_from_file_location(
        "allocation_test_launcher_gpu", _CONTROLLER.with_name("launch_live_test.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "12345\n")
    with pytest.raises(ValueError, match="occupied"):
        module.require_idle_gpu(2)
    with pytest.raises(ValueError, match="Only coordinated"):
        module.require_idle_gpu(0)


def test_historical_queue_waits_for_every_main_model(tmp_path, monkeypatch) -> None:
    import json

    monkeypatch.syspath_prepend(str(_CONTROLLER.parent))
    spec = importlib.util.spec_from_file_location(
        "allocation_final_queue", _CONTROLLER.with_name("run_final_eval_queue.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for arm, seed in module.ALL_MAIN[:-1]:
        output = tmp_path / module.name(arm, seed)
        output.mkdir()
        (output / "summary.json").write_text("{}")
        output.with_suffix(".launch.json").write_text(
            json.dumps({"registry_sha256": "registered", "status": "complete"})
        )
    assert not module.all_main_complete(tmp_path, "registered")
    arm, seed = module.ALL_MAIN[-1]
    output = tmp_path / module.name(arm, seed)
    output.mkdir()
    (output / "summary.json").write_text("{}")
    ledger = output.with_suffix(".launch.json")
    ledger.write_text(json.dumps({"registry_sha256": "registered", "status": "complete"}))
    assert module.all_main_complete(tmp_path, "registered")
    ledger.write_text(json.dumps({"registry_sha256": "another", "status": "complete"}))
    with pytest.raises(ValueError, match="another registry"):
        module.all_main_complete(tmp_path, "registered")
