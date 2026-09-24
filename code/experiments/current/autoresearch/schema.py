"""Pydantic schema for results/<NNN>_<slug>/params.json (frozen before a run) and
results.json (after).

`params.py new` writes a params.json with "TODO" placeholders; `Params.model_validate` rejects
it until every placeholder is replaced with a value of the right type, so the TODO check is the
type check. Extra keys are allowed everywhere (an agent adding a knob adds a key); missing or
mistyped ones fail.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Loose(BaseModel):
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------- params.json


class Arch(Loose):
    type: str
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int
    pos: str
    ctx: int
    norm: str
    act: str
    dropout: float
    tied_emb: bool
    n_params: int


class TokenizerCfg(Loose):
    codec: Literal["abs", "rel", "lean_seq"]
    maxn: int


class Optimizer(Loose):
    name: str
    lr: float
    schedule: str
    warmup_steps: int
    min_lr_frac: float
    wd: float
    betas: tuple[float, float]
    grad_clip: float


class Batch(Loose):
    size: int
    padding: str
    buckets: bool | str


class DataCfg(Loose):
    filter: str | None
    augmentation: str | None
    order: str
    val_slice: int


class Config(Loose):
    """What pretrain.py builds. Hand-filled by the agent; the patch is the ground truth."""

    arch: Arch
    tokenizer: TokenizerCfg
    optimizer: Optimizer
    batch: Batch
    data: DataCfg
    precision: str
    compile: bool | str
    init: str
    seed: int | list[int]  # scalar for historical runs; explicit list for multi-seed runs


class Pool(Loose):
    file: str | None = None
    n: int
    by_lines: dict[str, int]


class Budget(Loose):
    total_secs: int
    pretrain_secs: int
    grace_secs: int
    param_cap: int


class RlProtocol(Loose):
    rounds: int
    n_targets: int
    rollouts_per_target: int
    temp: float
    max_new_tokens: int
    ft_steps: int
    ft_bs: int
    ft_lr: float
    retain_pretrain_records: int
    max_proofs_per_thm: int
    rl_weight: int
    target_pool: Pool
    targets_sampled: Pool | None  # cached histogram is only available for seed 0


class EvalProtocol(Loose):
    metric: str
    attempts_per_theorem: int
    temp: float
    pool: Pool


class Protocol(Loose):
    """What the harness imposes. Mechanical: harness constants + pools/<fork-commit>.json."""

    fork_commit: str
    budget: Budget
    pretrain_data: Pool
    pretrain_heldout: Pool
    rl: RlProtocol
    eval: EvalProtocol
    holdout: Pool
    seed: int | list[int]


class Params(Loose):
    id: str = Field(pattern=r"^\d{3}(-[a-z]+)?$")  # e.g. 041-claude; no suffix before 041
    slug: str
    written_at: str | None
    provider: Literal["anthropic", "openai"]
    model: str
    base_commit: str
    harness_sha: str
    pretrain_sha: str
    patch_lines: int | None
    title: str
    description: str
    tags: list[str]
    hypothesis: str
    p_keep: float = Field(ge=0, le=1)
    expected_delta: str
    baseline_id: str | None
    baseline_metric: int | None
    config: Config
    protocol: Protocol
    migrated: bool = False
    patch_note: str | None = None


# ---------------------------------------------------------------- results.json

Status = Literal["keep", "discard", "crash", "timeout"]


class CurvePoint(Loose):
    secs: int
    step: int
    train_loss: float | None  # NaN (dumped as null) if no loss was logged before this point
    val_loss: float


class PretrainStats(Loose):
    secs: float | None = None
    n_params: int | None = None
    steps: int | None = None
    epochs: float | None = None
    last_loss: float | None = None
    heldout_greedy: float | None = None
    curve: list[CurvePoint] | None = None


class RlRound(Loose):
    round: int
    sample_acc: float
    new_proofs: int
    targets_solved: int
    lstar: int
    ge: dict[str, int]
    mix_rl_records: int
    samples_per_sec: int
    secs: int


class BinStat(Loose):
    solved: int
    n: int
    rate: float
    ci: tuple[float, float]


class EvalStats(Loose):
    solved: int | None = None
    n: int | None = None
    ge: dict[str, int] | None = None
    lstar: int | None = None
    heldout_greedy: float | None = None
    max_pruned: int | None = None
    by_bin: dict[str, BinStat] | None = None


class Results(Loose):
    commit: str
    recorded_at: str | None
    status: Status
    metric: int | None
    metric_rl: int | None
    harness_sha: str
    pretrain_sha: str
    rules: dict[str, int]
    gpu: str | None
    total_secs: int
    reason: str | None
    pretrain: PretrainStats
    rl_rounds: list[RlRound]
    eval: EvalStats
    holdout_metric: int | None
    surprise: str | None
    seed_metrics: dict[str, int | None] | None = None
    seed_metric_rl: dict[str, int | None] | None = None
    seed_status: dict[str, Literal["ok", "crash", "timeout", "missing"]] | None = None
    migrated: bool = False
