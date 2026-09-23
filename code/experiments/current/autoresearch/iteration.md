You run unattended: nobody will answer questions, so decide and proceed. Another agent runs
the same loop in parallel on the same branch (see program.md); start with `git pull --ff-only`.
Read AGENTS.md, .claude/notes/robbie-pretrain-workstream.md, .claude/notes/ideas.md, and
code/experiments/current/autoresearch/program.md. Complete exactly ONE pretraining
experiment under that protocol, then exit. The Python launcher starts the next session.

Keep the initial dataset, RL procedure, evaluation, budgets, and seed policy fixed.
Optimize downstream RL performance, not pretraining loss alone. Change only pretrain.py
and this experiment's registration/results/table; preserve unrelated work in the checkout.
Do not modify the launcher, runner, prompt, frozen files, or your permissions.

Robbie is excited about P01 (looped architectures) and P02 (Muon); these are interests,
not mandatory priorities. Include the chosen idea ID in the registration description;
check prior records to avoid accidental repeats.

Read previous successes and failures, choose one hypothesis, implement it, and fill the
preregistration. Use run_idea.py: it validates and commits/pushes the registration before
renting any pods. It must run in the foreground:
no nohup, tmux, background jobs, or direct pod provisioning. Preserve the
ND_AUTORESEARCH_SESSION environment variable; the runner uses it for cancellation ownership.

Choose keep/discard/crash/timeout and call record.py with that decision and your observation.
It writes the result, refreshes the table, keeps/restores pretrain.py, and commits/pushes the
outcome. Do not stage, commit, push, restore pretrain.py, or refresh the table yourself.
Finish with a brief summary. A published crash/timeout is a completed experiment, not a
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
