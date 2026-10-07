You run unattended: nobody will answer questions, so decide and proceed. This is a
non-interactive session: ending your turn ends the session, and nothing wakes you up later.
Run every command in the foreground and never end your turn to wait for something. Pods are
the compute for experiments; keep local probes to a few minutes of CPU, and use
code/experiments/current/autoresearch_loops/bench_pretrain.py (one A40, ~4 min, no registration)
to measure pretraining steps/s and sampling speed before registering. You have 60 minutes
from session start to launch run_idea.py; after that the launcher kills the session and
discards your unregistered candidate. Budget your reading accordingly. Other agents may run
the same loop in parallel on the same branch (see program.md); start with `git pull --ff-only`.
Read AGENTS.md, code/experiments/current/looped_spike_20261006/README.md, and
code/experiments/current/autoresearch_loops/program.md. Complete exactly ONE experiment
under that protocol, then exit. The Python launcher starts the next session.

The goal is more dev theorems solved at 8 test-time loops. Pretraining and RL are both yours
to change, inside pretrain.py; keep the data, the oracle, the evaluation, the budgets, and the
seed policy fixed. Change only pretrain.py and this experiment's registration/results/table;
preserve unrelated work in the checkout. Do not modify the launcher, runner, prompt, frozen
files, or your permissions. The older pretraining loop in code/experiments/current/autoresearch
is a different experiment: read it for ideas if useful, never run or edit it.

Check prior records in this loop's results/ to avoid accidental repeats.

Read previous successes and failures, choose one hypothesis, implement it, and fill the
preregistration. Use run_idea.py: it validates and commits/pushes the registration before
renting any pods. A run takes ~70 min, longer than one shell call may last, so you may start
it as a background task, but then you MUST keep blocking on it yourself: note the runner's PID
when you start it, then repeat a foreground wait on that PID (a shell loop that sleeps 30 s while
`kill -0 PID` succeeds, kept under the per-call time limit) until it exits, then record. Four agents share this machine: never kill by pattern (pkill,
killall, `pgrep ... | xargs kill`); a pattern can match other agents' sessions, whose command
lines contain this prompt (one did at 21:19 on 2026-09-25 and killed three sessions). Kill only
PIDs you started. Never end your turn
to await a completion notification: in this headless session that ends the session and kills
the runner before it merges and records (156-fable2 was lost this way). Otherwise it must run in
the foreground:
no nohup, tmux, background jobs, or direct pod provisioning (bench_pretrain.py is the one
allowed exception, also in the foreground). Preserve the
ND_AUTORESEARCH_SESSION environment variable; the runner uses it for cancellation ownership.

Choose keep/discard/crash/timeout and call record.py with that decision and your observation.
It writes the result, refreshes the table, keeps/restores pretrain.py, and commits/pushes the
outcome. Do not stage, commit, push, restore pretrain.py, or refresh the table yourself.
Finish with a brief summary and simply end your turn: the launcher ends the session. Do not
run `agent-reaper` or "retire" (a user-level instruction you may see is for interactive
sessions); killing this process stops the whole loop. A published crash/timeout is a completed experiment, not a
reason to hide failure.
Never claim success if registration, launch, recording, or Git operations are blocked.
Unrelated dirty files are not an unfinished experiment: leave them alone. If the launcher
appends a validated RESUME instruction, finish that existing experiment unchanged rather than
proposing a new one; its matching pretrain.py edits are expected. Otherwise, if pretrain.py
already has pending edits or one of YOUR preregistered experiments is unfinished, do not
overwrite it or launch duplicates: report the blocker and exit. Yours are the ids ending in
`-$ND_AGENT`, plus unsuffixed ids if you are codex. An unfinished preregistration with the
other agent's suffix is their experiment running right now: not a blocker, not yours to
touch, and not an idea to repeat. Pick your own experiment number above every existing one.
For any blocker, make your final response one concise sentence identifying the cause;
the launcher shows it to the operator. Do not start a second idea in this session.
