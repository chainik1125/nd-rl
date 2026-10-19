# exp22 — double the exploration budget on uncracked textbook schemas (prereg 2026-09-23 ~03:33 UTC)

**Question.** Across 8 textbook-stream runs only contraposition and export were ever cracked; T 1.2 (exp19) did not
crack a third. Is it a budget question? exp22 doubles the samples on uncracked-schema instances (k 32 -> 64 at T 1.2)
and pays for it with fewer generated-target samples (k 16 -> 12), whose frontier saturates anyway.

**Change vs incumbent.** `variants/exp22.py` = `variants/exp19.py` with tb_k 32 -> 64 and k 16 -> 12.

**Expected.** a 3rd schema cracked: 35 % chance (I expect it not to be). dev = 315 (range 280–345): fewer rounds and
fewer gen samples cost a little. **Falsified** (budget is not the missing piece) if only the usual 2 schemas crack.
**Seeds.** 0. **Stop rule.** One run; reruns only on crash.
