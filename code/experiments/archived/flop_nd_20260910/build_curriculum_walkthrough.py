"""Render a standalone report illustration from the two saved demonstration proofs.

This runs no prover, policy, training or evaluation. The displayed schedule outcomes
are supplied by the reader; they are not measured learning curves.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/minimal_seed_path_probabilities.json"
payload = json.loads(SOURCE.read_text())
payload["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>How a backward proof curriculum works</title>
<style>
:root{color-scheme:light;--ink:#172a36;--muted:#475c69;--line:#ccd8df;--blue:#175d91;--pale:#eaf3fa;--paper:#f8fafb;--teal:#086758}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.55 system-ui,sans-serif}
main{max-width:1060px;margin:0 auto;padding:42px 28px 50px}h1{font-size:38px;line-height:1.14;max-width:850px;margin:8px 0 16px;letter-spacing:-.7px}h2{font-size:24px;line-height:1.25;margin:0 0 12px}p{margin:10px 0 17px}.eyebrow{font-size:13px;font-weight:700;letter-spacing:1.3px;text-transform:uppercase;color:var(--teal)}.lead{max-width:890px;font-size:20px}.note{font-size:14px;color:var(--muted)}section{background:white;border:1px solid var(--line);border-radius:12px;margin-top:28px;padding:26px}a{color:var(--blue);text-underline-offset:3px}label{font-weight:650}select,button{font:inherit;border-radius:6px;border:1px solid #8298a5;background:white;color:var(--ink);padding:9px 12px}button{cursor:pointer}button:hover{background:var(--pale)}button.primary{background:var(--blue);color:white;border-color:var(--blue)}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #be6d00;outline-offset:3px}.controls{display:flex;align-items:center;flex-wrap:wrap;gap:12px;margin:16px 0}.range{width:100%;accent-color:var(--blue);margin:15px 0 5px}.ends{display:flex;justify-content:space-between;gap:15px;font-size:14px;color:var(--muted)}.facts{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0}.fact{border:1px solid var(--line);border-radius:8px;padding:15px}.big{display:block;font-size:30px;font-weight:700;line-height:1.2}.fact small{display:block;color:var(--muted);margin-top:5px}.formula{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere;font-size:16px}.theorem{padding:16px;background:#f0f5f6;border-left:4px solid var(--teal);margin:20px 0}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:15px}th{text-align:left;color:var(--muted);font-size:13px}th,td{padding:12px 10px;vertical-align:top;border-bottom:1px solid var(--line)}td:first-child{width:45px}td:last-child{width:195px}.supplied{background:#f2f5f6}.chosen{background:var(--pale)}.tag{display:inline-block;font-size:12px;font-weight:700;padding:3px 8px;border-radius:4px;background:#dae4e9;white-space:nowrap}.chosen .tag{background:#d5e9f8;color:#124a74}.hidden-formula{font-style:italic;color:var(--muted)}.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:14px;margin:12px 0}.legend span:before{content:"";display:inline-block;width:13px;height:13px;margin-right:6px;vertical-align:-1px;border:1px solid var(--line)}.legend .legend-given:before{background:#f2f5f6}.legend .legend-chosen:before{background:var(--pale)}.meter{position:relative;height:26px;background:#e6edf1;border-radius:4px;margin:18px 0 8px;overflow:hidden}.meter-fill{height:100%;background:var(--blue);width:0}.threshold{position:absolute;left:90%;height:100%;border-left:3px solid #934f00;top:0}.status{min-height:58px;padding:12px 15px;background:var(--pale);border-left:4px solid var(--blue);margin:15px 0}.history{font:14px/1.5 ui-monospace,monospace;color:var(--muted);min-height:24px;overflow-wrap:anywhere}.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}.two .fact{margin:0}.result{border-left:4px solid var(--teal);padding:3px 18px;margin-top:22px}.check{display:flex;gap:9px;align-items:center;font-size:14px}.check input{width:18px;height:18px;accent-color:var(--blue)}footer{font-size:14px;color:var(--muted);margin-top:28px}code{font-size:.9em}details{margin-top:18px}summary{cursor:pointer;font-weight:650}.hash{word-break:break-all;font-size:12px}
@media(max-width:650px){main{padding:25px 16px}h1{font-size:31px}.lead{font-size:18px}section{padding:19px 16px}.facts{grid-template-columns:1fr}.two{grid-template-columns:1fr}.big{font-size:27px}td:last-child{width:auto}.formula{font-size:14px}td,th{padding:10px 7px}.ends{font-size:12px}.table-wrap table,.table-wrap tbody{display:block}.table-wrap thead{display:none}.table-wrap tr{display:grid;grid-template-columns:28px minmax(0,1fr);border-bottom:1px solid var(--line);padding:10px 4px}.table-wrap td{border-bottom:0;padding:4px 6px}.table-wrap td:first-child{grid-column:1;grid-row:1 / span 2;width:auto;font-weight:700}.table-wrap td:nth-child(2){grid-column:2}.table-wrap td:last-child{grid-column:2;width:auto}.controls select{max-width:100%}}
@media print{body{background:white}main{max-width:none;padding:0}section{break-inside:avoid}button,.range{display:none}a{color:inherit}footer{font-size:11px}}
</style>
</head>
<body><main>
<div class="eyebrow">FLoP → natural deduction · interactive report illustration</div>
<h1>Teach the end of a proof, then remove the supplied deductions.</h1>
<p class="lead">A symbolic prover supplies legal inferences. A learned policy practices completing a verified proof from progressively earlier states. The same policy can later make many more decisions on a longer theorem.</p>
<p class="note">The proofs below are the actual two six-line seeds used in the minimal symbolic study. This page illustrates the mechanism; it does not run a model or display a measured learning curve.</p>

<section aria-labelledby="prefix-heading">
<h2 id="prefix-heading">1. Choose how much of the seed proof the learner must complete</h2>
<div class="controls"><label for="seed">Demonstration</label><select id="seed"><option value="0">Seed 1 · left, right, right, left, left</option><option value="1">Seed 2 · right, left, left, right, right</option></select></div>
<div class="theorem"><strong>Prove P from the premise</strong><div class="formula" id="premise"></div></div>
<label for="stage">Example deductions remaining: <output id="remaining-label" for="stage">1</output></label>
<input class="range" id="stage" type="range" min="1" max="5" value="1" step="1">
<div class="ends"><span>Near the end: one deduction to find</span><span>Original theorem: five deductions to find</span></div>
<div class="facts"><div class="fact"><span class="big" id="supplied-count">4</span><small>deductions supplied by prefix replay</small></div><div class="fact"><span class="big" id="chosen-count">1</span><small>deductions in the example completion</small></div><div class="fact"><span class="big" id="path-probability">4.00%</span><small>uniform probability of this exact completion path</small></div></div>
<p class="note">The probability uses the saved legal-action counts on this demonstration path. Other valid paths may exist. It is not the learner's success probability.</p>
<label class="check"><input id="show-example" type="checkbox" checked> Show the verified example completion for illustration</label>
<div class="legend"><span class="legend-given">Registered premise or replayed deduction</span><span class="legend-chosen">The learner must choose a completion</span></div>
<div class="table-wrap"><table><thead><tr><th>Line</th><th>Formula and justification</th><th>Role in the episode</th></tr></thead><tbody id="proof-lines"></tbody></table></div>
<p class="note">Only newly sampled actions enter the actor's PPO loss. The displayed example completion is not supplied to the policy. A different valid completion also earns success. At evaluation, no deduction is replayed; the actual premise is still registered.</p>
</section>

<section aria-labelledby="schedule-heading">
<h2 id="schedule-heading">2. Practice until the schedule removes one more supplied deduction</h2>
<p>Record hypothetical episode outcomes. Each result updates the local estimate as <code>new = 0.6 × old + 0.4 × success</code>. When it exceeds 0.9, practice moves one deduction earlier and the estimate resets.</p>
<div class="controls"><button class="primary" id="success" type="button">Record success</button><button id="failure" type="button">Record failure</button><button id="restart" type="button">Restart near the end</button></div>
<div class="meter" role="meter" aria-label="Local success estimate" aria-valuemin="0" aria-valuemax="1" aria-valuenow="0" id="ema-meter"><div class="meter-fill" id="ema-fill"></div><div class="threshold" title="Advance only above 0.9"></div></div>
<div class="ends"><span>Estimate: <strong id="ema-value">0.00000</strong></span><span>Advance above 0.90000</span></div>
<div class="status" role="status" id="schedule-status">Five consecutive successes from zero are enough to move one deduction earlier.</div>
<div class="history" id="history" aria-label="Hypothetical episode history"></div>
<p class="note">These are the released EMA settings used by the minimal symbolic study. Its schedule also forces advancement after more than 20,000 episodes at a level. This illustration shows the success-driven path. Failures lower the estimate without moving practice to an easier prefix. Changing the slider or seed resets the illustrative estimate.</p>
</section>

<section aria-labelledby="horizon-heading">
<h2 id="horizon-heading">3. Separate training difficulty from execution length</h2>
<p>The minimal experiment caps training at six written lines. Evaluation can let the learned scorer choose many more inferences. Each choice uses the updated symbolic state.</p>
<label for="depth">Illustrative atomic-projection certificate: <output id="depth-label" for="depth">129</output> lines</label>
<input class="range" id="depth" type="range" min="6" max="129" value="129" step="1">
<div class="ends"><span>6 lines</span><span>129 lines</span></div>
<div class="facts two"><div class="fact"><span class="big" id="inference-count">128</span><small>symbolic inference choices, excluding the premise</small></div><div class="fact"><span class="big" id="token-count">34,055</span><small>response tokens in the explicit certificate codec</small></div></div>
<p class="note">For the specified atomic-distractor references, d eliminations require <code>2d² + 10d + 7</code> response tokens because successive lines copy formulas. These are algebraic serialization costs, not shortest-proof bounds or new model evaluations. Legal token masking alone still leaves those tokens in the current LM's context.</p>
<div class="result"><strong>What the completed experiment found</strong><p>At 129 lines, the two curriculum policies solve 1/16 and 2/16 statements in a separate, later-added challenge. Those successes cover two atomic-projection statements, both concluding P. On the original 640-statement structured final set, curriculum scores 64 and 79 greedily, while cloning scores 74 and 83. Longer successful executions coexist with strong cloning and narrow generalization.</p></div>
</section>

<footer><p><a href="paper_walkthrough.md">Full paper walkthrough</a> · <a href="method_and_reproduction.md">Method and runnable commands</a> · <a href="sprint_report.md">Complete findings</a> · <a href="https://arxiv.org/pdf/1905.13100v2">FLoP paper</a></p>
<details><summary>Illustration provenance</summary><p>Seed proofs and path counts: <a href="minimal_seed_path_probabilities.json">saved source artifact</a>. No external scripts, network requests, training or model inference are used by this page.</p><p class="hash" id="source-hash"></p></details></footer>
</main>
<script id="saved-data" type="application/json">__SAVED_DATA__</script>
<script>
"use strict";
const saved = JSON.parse(document.getElementById("saved-data").textContent);
const el = id => document.getElementById(id);
let estimate = 0;
let episodeHistory = [];
const stage = () => Number(el("stage").value);
const notation = text => text.replaceAll("&", "∧");
function renderProof() {
  const seed = saved.sources[Number(el("seed").value)];
  const remaining = stage(), supplied = 5 - remaining;
  const lines = seed.body.split(";").map(x => x.trim()).filter(x => /^N\d+ /.test(x));
  el("premise").textContent = notation(lines[0].replace(/^N1 /, "").split(" : ")[0]);
  el("remaining-label").textContent = remaining;
  el("supplied-count").textContent = supplied;
  el("chosen-count").textContent = remaining;
  const probability = seed.candidate_counts_on_reference_path.slice(supplied).reduce((p, n) => p / n, 1);
  el("path-probability").textContent = probability >= 0.0001 ? (probability * 100).toFixed(2) + "%" : probability.toExponential(2);
  const body = el("proof-lines"); body.replaceChildren();
  lines.forEach((line, i) => {
    const provided = i <= supplied;
    const tr = document.createElement("tr"); tr.className = provided ? "supplied" : "chosen";
    const index = document.createElement("td"); index.textContent = i + 1;
    const content = document.createElement("td");
    if (provided || el("show-example").checked) {
      const formula = document.createElement("div"); formula.className = "formula";
      formula.textContent = notation(line.replace(/^N\d+ /, "").split(" : ")[0]);
      const reason = document.createElement("div"); reason.className = "note";
      const rawReason = line.split(" : ")[1];
      reason.textContent = rawReason.replace("ANDE1", "Left conjunction elimination,").replace("ANDE2", "Right conjunction elimination,").replace("PR", "Premise");
      content.append(formula, reason);
    } else { content.className = "hidden-formula"; content.textContent = "Completion to be chosen by the learner"; }
    const role = document.createElement("td"); const tag = document.createElement("span"); tag.className = "tag";
    tag.textContent = i === 0 ? "Premise" : provided ? "Replayed prefix" : "Choose an inference";
    role.append(tag); tr.append(index, content, role); body.append(tr);
  });
}
function renderEstimate() {
  el("ema-value").textContent = estimate.toFixed(5);
  el("ema-fill").style.width = (100 * estimate) + "%";
  el("ema-meter").setAttribute("aria-valuenow", String(estimate));
  el("history").textContent = episodeHistory.length ? "Outcomes: " + episodeHistory.slice(-24).join(" · ") : "No hypothetical episodes recorded.";
}
function resetEstimate(message) {
  estimate = 0; episodeHistory = []; renderEstimate();
  el("schedule-status").textContent = message;
}
function recordOutcome(success) {
  estimate = 0.6 * estimate + 0.4 * Number(success);
  episodeHistory.push(success ? "success" : "failure");
  if (estimate > 0.9 && stage() < 5) {
    const reached = estimate.toFixed(5);
    el("stage").value = stage() + 1;
    renderProof();
    resetEstimate("Estimate " + reached + " exceeded 0.9. One supplied deduction was removed; the example now has " + stage() + " deductions to complete. The local estimate resets to zero.");
  } else {
    renderEstimate();
    el("schedule-status").textContent = stage() === 5 && estimate > 0.9
      ? "Success is reliable in this hypothetical sequence at the original-theorem start. No supplied deductions remain to remove."
      : success ? "Success raises the local estimate. Advance requires an estimate strictly above 0.9."
      : "Failure lowers the local estimate. This schedule keeps the current starting state.";
  }
}
function renderHorizon() {
  const lines = Number(el("depth").value), deductions = lines - 1;
  el("depth-label").textContent = lines;
  el("inference-count").textContent = deductions.toLocaleString("en-US");
  el("token-count").textContent = (2 * deductions * deductions + 10 * deductions + 7).toLocaleString("en-US");
}
el("stage").addEventListener("input", () => {renderProof(); resetEstimate("Manual illustration level selected. The local estimate starts at zero.");});
el("seed").addEventListener("change", () => {renderProof(); resetEstimate("Demonstration changed. Each theorem has its own curriculum in training; this illustration resets its estimate.");});
el("show-example").addEventListener("change", renderProof);
el("success").addEventListener("click", () => recordOutcome(true));
el("failure").addEventListener("click", () => recordOutcome(false));
el("restart").addEventListener("click", () => {el("stage").value = 1; renderProof(); resetEstimate("Five consecutive successes from zero are enough to move one deduction earlier.");});
el("depth").addEventListener("input", renderHorizon);
el("source-hash").textContent = "Source SHA-256: " + saved.source_sha256;
renderProof(); renderEstimate(); renderHorizon();
</script>
</body></html>
'''
output = ROOT / "reports/curriculum_walkthrough.html"
output.write_text(template.replace("__SAVED_DATA__", json.dumps(payload).replace("<", "\\u003c")))
print(json.dumps({"output": str(output), "source_sha256": payload["source_sha256"], "bytes": output.stat().st_size}))
