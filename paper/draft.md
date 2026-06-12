# Fixing What Isn't Broken

\begin{center}
\textbf{Faizan Ahmed} \\
Headstarter \\
\texttt{faizan@theheadstarter.com}
\end{center}

## Abstract

Coding agents over-edit: they rewrite code that already passes its tests. Is the failure one of *representing* the pass/fail evidence, *reading it out*, or *acting on it*? We separate these three quantities and track them across the Qwen2.5-Coder family (1.5B–32B), with cross-model evidence from CodeGemma-7B and DeepSeek-Coder-1.3B, on SWE-bench-Verified-derived paired buggy/fixed prompts. **(1) A universal causal mechanism.** A direction at one late action-position site linearly encodes the pass/fail verdict and, under activation patching, causally moves the edit-vs-do-nothing margin at *every* Qwen scale; the site sits at relative depth 0.86–0.96 (a relative-depth law that drifts deeper with size, and width, not just depth, pushes it deeper), and its causal effect grows ~18× from 1.5B (+0.65 logits) to 32B (+11.7). **(2) Prior-gated, non-monotonic behavior.** Whether this evidence changes the *action* is not a function of scale: the do-nothing rate on passing prompts is 0/11/0/0/5% across 1.5B–32B, gated by whether the model's first-token action prior is edit-dominant. A binary {edit, noop} menu unmasks abstention for the view-prior 7B (0→12%), and rank-1 steering of the evidence direction causally flips the action to do-nothing on passing prompts (reaching a majority at high amplification, and flipping passing prompts before failing ones). The evidence is action-capable but suppressed at the operating point by the prior; the original 0%-at-1.5B finding is a special case, not a scaling law. **(3) A deployable edit-veto.** In a minimal multi-turn loop, over-editing is real and severe (3B–32B edit already-passing code 75–100% of the time, and having the passing transcript in context does not reduce it); used as an edit-veto, the internal pass/fail direction cuts over-editing 2–5× (e.g., 88%→20% at 3B, held-out) at high recall when the evidence is in context. Controls bound the direction: it tracks transcript text, not code correctness (three models), is near chance with no transcript, and a regex baseline matches it on the formats tested, so its value is mechanistic and as a cheap internal veto, not as a better transcript classifier. The throughline: pass/fail evidence is universally represented and causally wired to the edit/noop decision, its behavioral expression is gated by a graded, non-monotonic action prior, and the mechanism is deployable against real over-editing.

## 1. Introduction

Coding agents often over-edit: they "fix" code that already passes its tests, or keep acting when the right move is to stop. We ask whether this reflects a failure to *represent* the pass/fail evidence, to *read it out* into the action decision, or to *act on it*. Black-box action traces conflate three quantities: (a) whether pass/fail evidence is *available* in the residual stream, (b) whether it is *causally read out* into the `edit − noop` decision, and (c) the *final action* the model takes. Our central finding is that (a) and (b) hold **universally** across the Qwen2.5-Coder family (1.5B–32B): the evidence is represented and causally wired to the decision at every scale, while (c), whether it changes the action, is **non-monotonic and gated by the model's action prior**, not by scale. The mechanism is the constant; the behavior is the variable. Recent black-box reports characterize related action biases [@shang2026tebench; @fixedbench2025]; we open the mechanism they leave.

We study this first in a *static, single-turn* prompt over 499 SWE-bench-Verified-derived paired buggy/fixed prompts per model: each prompt pairs an issue, an oracle-localized 80-line code window, a synthesized pytest transcript, and a five-action menu (`view`/`grep`/`test`/`edit`/`noop`) scored at one action token (full ingestion in App. G.1). We then study a minimal *multi-turn agent loop*. On the static substrate the do-nothing (`noop`) rate on passing prompts is **non-monotonic** across scale: 0/11/0/0/5% at 1.5B/3B/7B/14B/32B, and is nonzero exactly when the first-token action prior is **edit-dominant** (3B, 32B), and zero when the prior favors investigation (`view`/`grep`; 7B, 14B). Position-balanced and abstract-label menu controls rule out a last-position artifact. A binary {edit, noop} menu unmasks abstention for the view-prior 7B (0→12%), and rank-1 steering of the evidence direction causally **flips** the action to abstention on passing prompts (reaching a majority at high amplification, flipping passing prompts before failing ones), so the evidence is *action-capable* but suppressed at the operating point by the prior. The original cross-model static result, that at 1.5B a strong `grep` prior keeps abstention at 0%, with reported-cell evidence on CodeGemma and DeepSeek, is thus the small-model, investigation-prior corner of a graded picture, not a scaling law.

Across the family, paired bidirectional residual patching localizes the readout to a late action-position site at every scale, whose relative depth follows a 0.86–0.96 law (drifting deeper with size; the 7B width control, same 28 layers as 1.5B but peaking three layers deeper, shows width, not just depth, pushes it deeper) and whose causal effect grows ~18× (+0.65→+11.7 logits). A contradictory-transcript control (three models) shows the direction tracks transcript *text*, not code semantics, and a regex or bag-of-words baseline matches it on the formats tested, so the static monitor's value is mechanistic, not as a better classifier. Finally, in a multi-turn loop the represented evidence predicts real, severe over-editing (3B–32B edit already-passing code 75–100% of the time), and used as an internal edit-veto the pass/fail direction cuts it 2–5× at high recall: the deployable payoff that a static text-baseline comparison alone cannot establish.

**What this paper does and does not show.** To keep the claims and
their boundary explicit:

*Shows.* (i) a **universal** causal pass/fail readout at the action
position across Qwen 1.5B–32B, a relative-depth law and an ~18× growth
in causal effect with scale; (ii) the behavioral expression is
**non-monotonic and prior-gated**, with rank-1 steering causally
*flipping* the action to abstention on passing prompts and a binary
{edit, noop} menu unmasking abstention for the view-prior 7B; (iii) in a
multi-turn loop, **severe over-editing** of already-passing code and a
held-out **edit-veto** that cuts it 2–5× at high recall.

*Does not show.* (i) real test execution: the agent loop uses
*simulated* observations (synthesized transcripts) over N=40 tasks with
greedy menu-constrained action selection, not a sandboxed SWE-bench run;
(ii) operational superiority over text parsing on clean transcripts: a
regex matches the static monitor (App. G.8); (iii) fully
multiple-testing-corrected localization at *every* scale: the
cross-family (CodeGemma/DeepSeek) cells are reported-cell, and the
large-model peak layers use coarse grids that may undershoot the exact
layer; (iv) free-form agentic behavior: actions are scored over a fixed
menu.

Our contributions, as three acts on a static-study foundation (bootstrap CIs and held-out estimates in App.; synthesis in Fig. \ref{fig:scaling}):

**Act I: A universal causal mechanism + depth-scaling law.** Across Qwen2.5-Coder 1.5B–32B, paired bidirectional residual patching localizes a causally-used pass/fail readout at a late action-position site at *every* scale (peaks L24/L32/L27/L44/L60; relative depth 0.857/0.889/0.964/0.917/0.938; bootstrap-stable, Wilcoxon $p<10^{-9}$), a relative-depth law that drifts deeper with size, with a fine-swept 7B width control (same 28 layers as 1.5B but peaking at the *final* layer L27 not L24) showing width, not just depth, pushes the site deeper. The causal effect grows ~18× (+0.65→+11.7 logits), and the discriminability-vs-causal-use gap (max-AUC layer $\neq$ causal layer) grows with scale.

**Act II: Prior-gated, non-monotonic behavior, with causal control.** The do-nothing rate on passing prompts is non-monotonic, 0/11/0/0/5% (1.5B→32B), gated by whether the first-token action prior is edit-dominant. A binary {edit, noop} menu unmasks abstention for the view-prior 7B (0→12%); rank-1 steering of the evidence direction causally *flips* the action to do-nothing on passing prompts (majority at high amplification, passing before failing, an evidence-tracking asymmetry). Unlike the refusal direction [@arditi2024refusal], which *controls* behavior, this evidence direction is *action-capable but suppressed at the operating point by the prior* (§5.2, §5.8).

**Act III: A deployable edit-veto.** In a minimal multi-turn loop, over-editing is real and severe (3B–32B edit already-passing code 75–100%; the passing transcript in context does not reduce it), and the agent often fails to gather the evidence (3B tests 2%; for 7B, testing halves over-editing, 47% vs 100%). Used as an edit-veto, the internal pass/fail direction cuts over-editing 2–5× (88%→20%, held-out) at 60–94% recall when the evidence is in context: the operational use that the static text-baseline parity does not provide.

**Foundation: the paired substrate, frozen monitor, and cross-model controls** that Acts I–III build on are detailed next (the original 1.5B + CodeGemma + DeepSeek static study; Acts I–II generalize its localization and behavior across scale, and Act III puts the monitor in a live loop):

1. **A represented-evidence-vs-action dissociation (the 1.5B special case).** We separate three quantities that black-box action traces conflate: (a) availability of pass/fail evidence in the residual stream, (b) its causal readout into the `edit − noop` margin, and (c) the final action, and show (a) and (b) hold on Qwen while (c) abstains 0% *at 1.5B specifically*; Acts I–II show (c) is graded and prior-gated across scale. Unlike the refusal direction [@arditi2024refusal], which *controls* behavior, this evidence direction *moves the decision variable without (at 1.5B) changing the action* (§5.2, §5.5).
2. **A paired buggy/fixed substrate (toy + SWE-bench-Verified-derived).**
 49 LLM-generated Python tasks plus 499/497/499 paired prompts
 (Qwen/CodeGemma/DeepSeek) from SWE-bench Verified
 [@jimenez2024swebench], an oracle window plus a synthesized pytest
 transcript per condition (App. B.1, G.1); identical static prompts, not an
 agent loop. The `edit − noop` logit margin is the scalar behavioral signal.
3. **Causal localization on Qwen, with cross-model reported-cell evidence.**
 Paired residual patching at Qwen L24/pos −1 (`resid_pre`) moves the
 `edit − noop` margin (mean F→B +0.648 logits over 49 toys, 100% positive;
 recovers ~98% of the +0.659 buggy/fixed gap; permutation null
 *p* < $10^{-4}$ on the 43-task subset). Rank-1 steering with $v_{\rm raw}$
 (unit $u_{\rm tx}$, the transcript-evidence direction from the
 fixed-minus-buggy toy contrast; legacy name `v_noop`) reproduces it.
 A deterministic 200-pair SWE-derived peak-cell check on Qwen supports
 transfer of the bidirectional submargin shift at L24/pos −1; the high-AUC
 wrong-layer control (L12/pos −1) is an order of magnitude smaller
 causally, and the wrong-position control (L24/pos −8) is near zero
 (§4.1, Table \ref{tab:swe-peak-patching}). Reported-cell
 patching gives analogous submargin effects on CodeGemma (L26)
 and DeepSeek (L22); the DeepSeek action-margin result
 is a first-subword proxy because canonical `noop` is multi-token under
 its tokenizer.
 Full bidirectional sampled grid and the max-statistic permutation null
 are Qwen-only; CodeGemma has an exploratory one-way F→B layer×position
 heatmap (App. D) plus reported-cell paired values; DeepSeek is
 reported-cell paired only. The no-transcript `code` variant has no Qwen peak, so the
 causal claim is specific to code-plus-transcript prompts (§4; App. D).
4. **A monitor that reads transcript evidence, bounded by controls and text
 baselines.** A frozen one-dot-product projection onto each model's direction
 $u_{\rm tx}$ (at its §4.3 reported cell) separates the 499/497/499 paired prompts
 at ROC-AUC **0.989 / 0.950 / 0.888** with no real-task supervision. A
 contradictory-transcript control (all three models) shows the score follows
 the transcript label, not code correctness; a trivial regex (AUC 1.000) and a
 bag-of-words baseline match or beat it on the clean and noisy/degraded
 pytest-style transcript transforms tested here. On
 Qwen the readout survives one deterministic NL paraphrase (AUC 0.995), but
 keyword/BoW baselines also reach 1.0, so this is not a semantic abstraction, and in a stricter exploratory Qwen disjoint-vocabulary held-out-template test
 it reaches AUC 0.943 while literal and train-vocabulary keyword baselines are
 at chance and a train-fit BoW baseline reaches pooled AUC 0.750
 (template-based, App. G.17). The CodeGemma and DeepSeek reported-cell
 paraphrase follow-up is negative, so this held-out robustness is
 Qwen-specific. It is near chance with no
 transcript (AUC ≤ 0.52). Post-hoc cross-format cells are selection-biased
 (App. G.13–G.14).
5. **Action-menu and five-action controls on the behavior.** Position-balanced,
 binary, and abstract-label menus show the 0% first-token abstention result
 is not primarily a last-position artifact on Qwen and CodeGemma; a
 single-token DeepSeek rerun reveals a first-position bias. Qwen five-action
 steering and discrete patching, plus a DeepSeek single-token reported-cell
 follow-up, show that interventions move non-abstention action competition
 and the `edit − abstain` submargin while abstention remains noncompetitive.
 The robust result is submargin modulation, not abstention induction
 (§5.5–5.6).
6. **Temporal separation (Qwen).** With the transcript pushed two turns upstream
 behind condition-neutral turns, the frozen $u_{\rm tx}$ still discriminates
 the failing-transcript (buggy) from the passing-transcript (fixed) condition
 at **AUC 0.807** (vs 0.989 adjacent; 0.509 no-transcript
 control), beating a stateless turn-local regex (0.500) but not a full-history
 or stateful parser, mechanistic evidence of a carried-forward within-context
 representation, not deployment superiority. The first-token argmax is
 identical across conditions, the same represented-evidence-versus-action gap
 behind the 0% result (§5.3, App. G.15).

Exploratory sparse-autoencoder (SAE) decomposition is in Appendix H (the direction
is geometrically dense in the learned basis; small orthogonal-matching-pursuit-selected
feature subsets partially affect the margin but are seed-fragile); it is not a core contribution.

Together, these results recast the static "evidence represented but not
acted on" finding as one corner of a graded, scale-spanning picture.
Pass/fail evidence is **universally represented and causally wired** to
the edit/noop decision, the mechanism strengthens ~18× with scale and
follows a relative-depth law, while its **behavioral expression is gated
by a non-monotonic action prior** that steering and a binary menu can
override, so abstention is action-capable but suppressed at the operating
point rather than absent. The monitor is not a code-correctness detector
and not a better transcript classifier than text parsing; its value is
mechanistic *and*, read internally in a live loop, **deployable as an
edit-veto** that cuts real over-editing 2–5×. The throughline is where and
how pass/fail evidence enters the action computation, why that represented
evidence is gated out of first-token behavior at some scales, and how to
read it back out to change behavior.

**Paper structure.** §2–§6 cover the technical core: related work,
methods, causal localization, monitor controls, and discussion. Separate
sections cover ethics, LLM use, and broader impacts. Appendix G
contains the major specificity, text-baseline, paraphrase,
temporal-separation, and cross-model controls; Appendix H contains the
exploratory SAE decomposition; Appendix J gives reproduction details.

## 2. Related Work

**Coding-agent action bias.** Two recent benchmarks document over-
editing as a black-box phenomenon. @shang2026tebench's TEBench is a project-level test-evolution
benchmark that catalogs *test-evolution* tasks, Test-Breaking, Test-Stale,
and Test-Missing cases, and asks agents to identify and update tests;
the stale-test subset is one piece of the broader test-evolution
problem. @fixedbench2025 evaluate coding agents on **stale bug
reports** where no code changes are required, and report undesirable
code changes in 35–65% of cases, depending on model/harness, directly
the behavioral phenomenon a no-transcript no-edit detector would target. Where these benchmarks measure
*what the agent does*, we ask *where in the residual stream the
pass/fail test-log evidence is read out into the action decision*. Our 499
paired prompts are derived from SWE-bench Verified
[@jimenez2024swebench] (full ingestion in App. G.1; this is
**not** the agent-loop evaluation that benchmark is normally used
for, and we use the phrase "SWE-bench-Verified-derived static
paired prompts" consistently for that reason); the SWE-agent
scaffold [@yang2024sweagent] and Agentless decomposition
[@xia2024agentless] frame the localization→repair→validation
pipeline that a future agent-loop veto monitor would sit at the
localization/validation boundary of.

**Mechanistic interpretability methods.** Activation patching
[@meng2022rome; @wang2023ioi; @heimersheim2024patching;
@zhang2024patchingbestpractices] and rank-1 activation steering
[@turner2023actadd; @rimsky2024caa; @huang2025manifold] are standard
tools for localizing causal sites and validating directions;
@elhage2021mathematical frames the circuit-level view; the
representation-engineering line [@zou2023representation] situates
direction-level read/control under an alignment umbrella. Most
directly relevant, @arditi2024refusal show that refusal in chat
models is mediated by **a single residual-stream direction**;
our contrast direction is structurally similar (one contrastive
direction moving a logit margin) but where the refusal direction *controls* the
behavior, ours *moves the `edit/noop` submargin without changing the action* and
tracks explicit pass/fail transcript evidence rather than an abstention belief, a
represented-evidence-vs-action dissociation (§5.2). @hewitt2019probes
caution that high-AUC probes do not establish a feature is
*used*; we accordingly treat probes as a necessary-but-not-sufficient check and rely on patching and steering for causal
claims.

**SAE feature circuits.** @bricken2023monosemanticity and
@templeton2024scaling established dictionary-learning
interpretability at scale; @gao2024scaling introduced the TopK SAE
we use; @marks2024sparsefeaturecircuits pioneered the
sparse-feature-circuit framing we extend with OMP-then-ablate;
@conmy2023acdc gives the automated circuit-discovery context;
@lieberum2024gemmascope provides pretrained SAE suites we did not
exercise here. @tahimic2025codecorrectness train SAEs on code-LLM
residuals for code correctness; we study the agent's *decision to
mutate the repository*, which is upstream of correctness assessment.
Per-paper detail in Appendix A.

## 3. Methods

We work on a paired buggy/fixed substrate of 49 LLM-generated toy
Python tasks and 499/497/499 SWE-bench-Verified-derived paired prompts
(App. G.1), both ingested into an identical static agent-prompt
format with a five-action vocabulary (`view`, `grep`, `test`,
`edit`, `noop`). These names are single-token in the exact scored context
(the first token after the final `Action: `) under Qwen and CodeGemma; under
DeepSeek, `grep` and `noop` are two tokens, so DeepSeek action-logit readouts
on those names are first-subword proxies (audit in §\ref{sec:tok-audit}). The scalar of interest is the
$m \;\triangleq\; \ell_{\text{edit}} - \ell_{\text{noop}}$ logit
margin at the action token. **Sign convention.** Throughout we
report $m_{\text{buggy}}$ vs $m_{\text{fixed}}$ separately or their
gap; the *baseline* fact is $m_{\text{buggy}} > m_{\text{fixed}}$
(buggy evidence raises edit preference), so positive
$\Delta = m_{\text{buggy}} - m_{\text{fixed}}$ means the residual
stream is tracking the buggy/fixed contrast in the expected
direction. **Directions.** Let
$v_{\rm raw}=\mu_{\rm fixed}-\mu_{\rm buggy}$ denote the raw
contrastive mean-difference vector at a residual-stream cell, and
let $u_{\rm tx}=v_{\rm raw}/\|v_{\rm raw}\|$ denote its
unit-normalized monitor direction. For a residual $h$ we report
two related scalars: the **projection** $p=h\cdot u_{\rm tx}$
(lower = failing-transcript evidence) and the **classifier
score** $s=-p=-h\cdot u_{\rm tx}$ (higher = failing-transcript
evidence). Additive steering uses $v_{\rm raw}$ unless otherwise
stated. Sign conventions (tables state whether they report $p$ or $s$):

| quantity | higher means |
|---|---|
| projection $p = h\cdot u_{\rm tx}$ | passing / fixed transcript evidence |
| score $s = -p$ | failing / buggy transcript evidence |
| margin $m = \ell_{\rm edit}-\ell_{\rm noop}$ | edit favored over noop |

We call the unit direction $u_{\rm tx}$ because the §5.2 controls
show it tracks pass/fail **transcript evidence** (the legacy name
`v_noop` reflects that it was first derived while studying the
`edit − noop` submargin). Because §5.2 shows that this direction tracks
transcript evidence rather than code semantics or an independent
no-edit decision, the legacy name `v_noop` should be read as a name
for the pass/fail-transcript monitor direction. Some figure
labels, artifact paths, and legacy prose
retain the name `v_noop`; throughout, this denotes the unit
monitor direction $u_{\rm tx}$ unless explicitly described as
the raw contrast vector $v_{\rm raw}$. Per-task tables call out
the direction explicitly (e.g. "F→B" denotes patching from a fixed
prompt into a buggy run). We use three complementary analyses on
the residual stream: probes (logistic regression at every
(layer, position), used only as availability checks), paired
activation patching (on Qwen, substituting FIXED→BUGGY and BUGGY→FIXED
across a sampled layer×position grid; on CodeGemma and DeepSeek,
applying the analogous paired intervention at the reported patch cells
and measuring the shift in margin), and rank-1 additive
steering (adding $\alpha\,v_{\rm raw}$ at the patching peak). The
causal interventions (patching and steering) act on `resid_pre`;
probes are reported separately as non-causal availability checks.
Full protocol, prompt variants, hook implementation, and model
details in Appendix B.

## 4. Results

**Behavioral setup.** Probes saturate at AUC ≈ 1.0 at every
(layer, position) under `code_tests` (Appendix C), confirming that the
information is widely available in the residual stream, but not
identifying where the action computation *uses* it. Behaviorally, only the `code_tests` variant
(issue + code + test transcript) produces a systematic
buggy/fixed action shift on Qwen and CodeGemma, the test
transcript is the only evidence level that drives the effect we localize next
(per-variant table for Qwen and CodeGemma in Appendix C;
DeepSeek's `code_tests` toy gap is reported in §4.3).

### 4.1 Causal patching

Substituting the FIXED residual into the BUGGY forward at
(layer × position) on Qwen yields a heatmap (Figure \ref{fig:mechanism}B) with a
clean late-layer concentration at the action position. The signal
is essentially zero in L0–L16 and ramps monotonically through
L18–L26, peaking at **L24/pos −1** with **mean shift +0.648
logits** (median +0.688, 100% of all 49 tasks positive in F→B).
This recovers ~98% of the all-49 clean B−F margin gap of +0.659
reported in the cross-model summary below. The bidirectional grid and the App. D permutation null
use the **43-task subset** for which both F→B and B→F grid
values were cached; on that subset the peak-cell F→B and B→F
mean shifts are **+0.69 and +0.64 logits** with 100% per-direction positivity. The bidirectional minimum
(min(F→B, B→F) per cell, then ranked) puts the same site at the
top, ruling out the trivial alternative that we are merely
erasing information. A `code`-variant negative control (no test
transcript) on Qwen shows no positive peak, F→B mean shift at L24/pos
−1 is +0.015 (median 0.000, 45% positive, essentially chance), confirming the site is a test-evidence-conditional readout.
Qwen patching heatmap, exploratory CodeGemma heatmap, and the Qwen negative control in Appendix D.

**SWE-derived peak-cell causal check (Qwen).** Paired residual patching
at three cells, **L24/pos −1** (Qwen causal readout site identified
on the toy grid), **L12/pos −1** (wrong-layer same-position; high-AUC
and decodable in §5.1 but causally weak under patching), and
**L24/pos −8** (wrong-position same-layer), on a deterministic
**N = 200** subset of the §5.1
SWE-bench-Verified-derived `code_tests` paired prompts. We use N = 200
because paired residual patching requires fresh patched forwards for
each cell; the subset is fixed before analysis and selected with seed 0
stratified by repository; the clean B−F margin gap on the subset is
+0.332 logits (scripts in App. J). Per-cell shifts are reported in
Table \ref{tab:swe-peak-patching} as mean with paired bootstrap 95\%
CI ($B = 10\,000$, seed 0), median, and per-task \% positive.

\begin{table}[t]
\caption{\textbf{SWE-derived peak-cell patching check on Qwen.} Paired
residual patching on a deterministic $N = 200$ SWE-bench-Verified-derived
subset (seed 0, stratified by repository). L24/pos $-1$ is the Qwen
causal readout site identified on the toy grid; L12/pos $-1$ is a
high-AUC wrong-layer same-position control; L24/pos $-8$ is a
wrong-position same-layer control.}
\label{tab:swe-peak-patching}
\centering
\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrrr}
\toprule
cell & F$\to$B mean [95\% CI] & F$\to$B med. & F$\to$B \% pos. & B$\to$F mean [95\% CI] & B$\to$F med. & B$\to$F \% pos. \\
\midrule
\textbf{L24 / pos $-1$} (causal readout) & $\mathbf{+0.314\,[+0.278,+0.351]}$ & $+0.312$ & $\mathbf{86\%}$ & $\mathbf{+0.311\,[+0.272,+0.349]}$ & $+0.312$ & $\mathbf{83\%}$ \\
L12 / pos $-1$ (wrong-layer same-pos.) & $+0.029\,[+0.018,+0.039]$ & $+0.000$ & $47\%$ & $+0.014\,[+0.004,+0.025]$ & $+0.000$ & $40\%$ \\
L24 / pos $-8$ (wrong-pos.\ same-layer) & $-0.001\,[-0.007,+0.006]$ & $+0.000$ & $23\%$ & $+0.004\,[-0.003,+0.010]$ & $+0.000$ & $27\%$ \\
\bottomrule
\end{tabular}}
\end{table}

Table \ref{tab:swe-peak-patching} shows a large bidirectional shift at
**L24/pos −1** (Wilcoxon one-sided *p* < $10^{-26}$ in both
directions; recovers ~95% of the subset's clean margin gap on F→B and
~94% on B→F), **L12/pos −1** an order of magnitude smaller despite its
higher discriminative AUC at the same position (§5.1; the wrong-layer
control reaches AUC 0.998 vs L24's 0.989), and **L24/pos −8**
essentially zero. *This is the clearest separation between
representation and causal use:* L12/pos −1 linearly separates
transcript labels even better than L24/pos −1, but patching it barely
moves the action submargin. The L24 site is therefore justified by
causal readout, not by maximal decodability.

All five action logits are logged: `noop` remains noncompetitive
(0/200 argmax under both patch directions). Discrete argmax changes,
when present, occur within the `grep`/`edit` pair. The direction of
this reallocation differs between toy and SWE-derived subsets (§5.6),
so we interpret the robust result as submargin modulation, not a stable
full-action policy. This is a peak-cell check on a deterministic
SWE-derived subset, not a full SWE-derived localization grid; it
supports transfer of the causal readout, not full cross-distribution
circuit identity. Artifacts and scripts are listed in App. J.

### 4.2 Steering with a single rank-1 direction

Using the §3 directions, $v_{\rm raw}=\mu_{\rm fixed}-\mu_{\rm buggy}$
at L24/pos −1 on Qwen has $\|v_{\rm raw}\| = 5.892$ (N=49).
**Additive steering adds $\alpha\,v_{\rm raw}$** to the residual
at the cell; sweeping α yields the dose-response in
Figure \ref{fig:mechanism}C: the mean `edit − noop` margin moves
smoothly and monotonically with α in both conditions, and α = +1
drops the buggy margin by 0.656 logits, essentially the full
behavioral gap on this scalar.
**A single contrastive direction is sufficient to steer the
`edit − noop` margin under our prompt distribution.** We do
**not** claim the information is uniquely one-dimensional or
absent from higher-rank subspaces: PCA1 at the same cell is
comparable (AUC 0.977 vs $u_{\rm tx}$'s 0.989; App. G.2), and
probes saturate at AUC ≈ 1.0 at every (layer, position) under
`code_tests` (App. C), so the information is broadly distributed
in residual space. What $u_{\rm tx}$ adds is a **labeled** direction
extracted by contrastive mean-difference that controls the
margin behaviorally. Per-task dose-response curves in Appendix E.

### 4.3 Cross-model evidence (3 model families)

Applying analogous reported-cell analyses to two additional models:

\begingroup
\setlength{\tabcolsep}{4pt}
\noindent\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrrrrr}
\toprule
model & layers & reported patch cell (layer, pos) & rel.\ depth &
 clean (B$-$F) margin gap & F$\to$B shift at cell & B$\to$F shift at cell &
 $\|v_{\rm raw}\|$ & toy N \\
\midrule
\mbox{Qwen2.5-Coder-1.5B-Inst} & 28 & L24, pos $-1$ & 0.857 &
 $+0.659$ & $+0.69$ & $+0.64$ & $5.89$ & 49 \\
\mbox{CodeGemma-7B-it} & 28 & L26, pos $-1$ & 0.929 &
 $+1.347$ & $+1.143$ & $+1.184$ & $6.181$ & 49 \\
\mbox{DeepSeek-Coder-1.3B-Inst}$^\dagger$ & 24 & L22, pos $-1$ & 0.917 &
 $+0.204$ & $+0.194$ & $+0.184$ & $12.255$ & 49 \\
\bottomrule
\end{tabular}}
\endgroup

$^\dagger$DeepSeek's canonical `noop` label is multi-token in the exact scored
context, so DeepSeek's `edit − noop` margins and patching effects above use the
first subword of `noop` as a proxy (§\ref{sec:tok-audit}). Projection-monitor
AUCs and contradictory-transcript score decompositions are unaffected by this
action-label tokenization issue, but DeepSeek action-level and margin-level
claims should be read as proxy analyses unless rerun with a single-token
abstention label such as `done`; a single-token DeepSeek rerun shows a
first-position bias (§5.5). The L22 reported cell was also selected under this
proxy action-margin analysis, so DeepSeek site-level claims should be read as
reported-cell evidence under proxy scoring rather than exact canonical-action
localization.

**Qwen 49/43 accounting (table-local).** For Qwen, the clean
B$-$F margin gap and $\|v_{\rm raw}\|$ use all 49 toys; the
F$\to$B/B$\to$F shift-at-cell columns use the 43-task bidirectional grid
subset (the subset for which both directions were cached). The
all-49 one-way F$\to$B estimate at the same cell is $+0.648$
logits (100% positive per-task; §4.1).

Column definitions: **clean (B$-$F) margin gap** = behavioral
$\ell_{\rm edit} - \ell_{\rm noop}$ logit-margin difference
between buggy and fixed on the toy substrate (mean over toy N
tasks); **F$\to$B shift at cell** and **B$\to$F shift at cell** = mean
$\Delta$-margin under residual-substitution patching at the reported
patch cell (toward closure of the gap, signed positive). For Qwen this
cell is the multiple-testing-corrected patching peak (full sampled
grid + max-statistic permutation null); for CodeGemma and DeepSeek
these are reported patch-cell checks (App. D for CodeGemma's
exploratory one-way F$\to$B grid), not full corrected localizations.
**$\|v_{\rm raw}\|$** = norm of the *raw* toy mean-difference
vector $v_{\rm raw} = \mu_{\rm fixed} - \mu_{\rm buggy}$ before
unit normalization; this equals the fixed-minus-buggy projection
gap along the unit direction $u_{\rm tx} = v_{\rm raw} /
\|v_{\rm raw}\|$ (the monitor direction, legacy shorthand
`v_noop`);
**toy N** = number of toy paired tasks the direction is derived
from (CodeGemma row uses the all-49 direction matching §5.1;
the 20-task responsive subset ($\|v\|=6.678$, mean B$-$F gap
$\approx +2$ logits on the responsive subset) is used only for
the §4.2 steering plot and the exploratory CodeGemma SAE in
App. H.6, where non-responsive toys diluted the dose-response
slope). The gap and
$\|v_{\rm raw}\|$ columns measure different quantities, Qwen's $5.89$ is $\|v_{\rm raw}\|$, not the behavioral
margin gap of $0.659$.

All three reported cells are at the action position; relative depths
cluster in [0.857, 0.929]. **CodeGemma**'s F$\to$B/B$\to$F shifts at its reported cell
are ~1.7× Qwen's, but per-task universality is bimodal (53–61%
of tasks recover ~2 logits, the rest near zero), motivating the
responsive-subset direction for §4.2 steering even though §5.1
uses the all-49 direction. **DeepSeek-1.3B**'s toy buggy-fixed
gap is tiny ($+0.20$, median 0.00; 16 of 49 toys have identical
margins under buggy and fixed), yet patching at L22/pos $-1$
recovers 90% of *that* small gap and DeepSeek's $v_{\rm raw}$
($\|v_{\rm raw}\|=12.255$) transfers to SWE-bench-Verified-derived tasks (§5.1). The same
patching protocol yields late action-position effects at the
reported cells on every model (reported-cell paired-patching, not full
multiple-testing-corrected localization); absolute
behavioral-saliency on toys varies by an order of magnitude
across models, a divergence we
revisit in §6.

![**Qwen causal localization and steering.** *(A)* Paired buggy/fixed toy substrate. *(B)* F→B residual patching peaks at L24/pos −1 on the `edit − noop` margin. *(C)* Rank-1 steering along the toy fixed-minus-buggy contrast direction reproduces the margin shift. CodeGemma steering is shown only as a responsive-subset comparison; Qwen is the fully localized result. Panels (B)–(C) establish the *available* and *causally read out* steps of the represented-evidence-vs-action gap; the final action does not change at 1.5B's natural operating point (0% `noop`; steering flips it, §5.8).](figures/main_mechanism.png){#fig:mechanism width=99%}

### 4.4 Scaling the causal mechanism across the Qwen family (1.5B–32B)

We extend the §4.1 localization from 1.5B to the full Qwen2.5-Coder-Instruct
ladder, 1.5B, 3B, 7B, 14B, 32B, holding the 49-toy and 499-SWE
substrates, the tokenizer, the chat template, and the five-action menu fixed, so
the *only* variable across runs is the forward pass (all five action names remain
single-token at every size; §\ref{sec:tok-audit}). Per size we run paired
bidirectional residual patching on the 49 toy `code_tests` pairs (coarse
layer-step 2–4, plus a fine layer-step-1 sweep on 7B) and take the peak as the
cell maximizing the bidirectional minimum $\min(\text{F}\!\to\!\text{B},
\text{B}\!\to\!\text{F})$ on the `edit − noop` margin.

**A causally-used pass/fail readout exists at every scale.** The peak is at the
action position (pos −1), in the late layers, bidirectionally significant
(Wilcoxon one-sided $p<10^{-9}$ on the 49 per-task F→B effects), and
bootstrap-stable (the peak layer is recovered in 92–100% of task resamples) at
every size:

| size | layers | peak (pos −1) | rel.\ depth | F→B [95% CI] | toy B−F gap | Wilcoxon $p$ |
|---|---:|---|---:|---|---:|---:|
| 1.5B | 28 | L24 | 0.857 | $+0.65$ | $+0.66$ | (paper) |
| 3B | 36 | L32 | 0.889 | $+1.00\,[+0.86,+1.15]$ | $+1.25$ | $<10^{-9}$ |
| 7B | 28 | L27 (fine) | 0.964 | $+2.77$ | $+2.76$ | $<10^{-9}$ |
| 14B | 48 | L44 | 0.917 | $+4.09\,[+3.53,+4.66]$ | $+4.89$ | $<10^{-9}$ |
| 32B | 64 | L60 | 0.938 | $+11.66\,[+11.07,+12.25]$ | $+14.14$ | $<10^{-9}$ |

The 7B peak is from a fine layer-step-1 sweep (L18–27); the coarse step-2 grid
undershot at L26/0.929. The `code`-variant negative control shows no peak at every
size, as at 1.5B, so the site is transcript-conditional.

**A relative-depth law that drifts deeper, and width, not just depth, pushes it
deeper.** The peaks sit at relative depth 0.857/0.889/0.964/0.917/0.938
(1.5B→32B), a late-layer band that creeps toward the final layer with size. The
cleanest test is the 7B **width control**: 7B has the *same 28 layers* as 1.5B but
~2.3× the width, and a fine layer-step-1 sweep (L18–27) shows its F→B effect
rising monotonically to the **final layer** (L27, relative depth 0.964, $+2.77$
logits), far deeper than 1.5B's L24 (0.857). The readout's location is therefore
not a pure function of relative depth; added width defers it toward the network's
end. (Coarse grids undershoot the exact layer, so the 3B/14B/32B peaks are likely
a touch deeper than reported.)

**Causal effect grows ~18× with scale.** The peak F→B shift grows monotonically
from $+0.65$ logits (1.5B) to $+11.7$ (32B), recovering ~80–100% of the toy
buggy–fixed gap at the coarse peak. The mechanism becomes far more behaviorally
salient with size.

**Discriminability $\neq$ causal use, and the gap grows.** A per-layer monitor
scan on 3B (each layer's own toy contrast direction projected onto the 499 SWE
residuals) peaks at L27 (ROC-AUC 0.977), as separable as 1.5B's best, while the
*causal* cell L32 reads only 0.867; the gap between the max-AUC layer and the
causal-use layer is $+0.110$ at 3B vs ~0.01 at 1.5B. Discriminative AUC does not
localize the mechanism (§5.1), and that divergence widens with scale.

![**Scaling synthesis (Qwen2.5-Coder).** *(A)* The causal readout's relative depth follows a late-layer law that drifts deeper with size; the 7B width control (fine-swept) peaks at the final layer. *(B)* The peak causal `edit − noop` effect grows ~18× (95% CIs). *(C)* The do-nothing rate on passing prompts is non-monotonic and prior-gated; a binary {edit, noop} menu unmasks abstention for the view-prior 7B. *(D)* In a live agent loop, over-editing of already-passing code is severe across scale and is cut by a held-out monitor edit-veto.](figures/scaling_synthesis.png){#fig:scaling width=99%}

## 5. From a Static Monitor to What It Actually Reads

The `stale_misleading` and `stale_flaky` variants probe what
happens when evidence disagrees with itself (Qwen, N = 49 each):

| variant | argmax = edit | argmax = noop | argmax = grep | projection $p=h\cdot u_{\rm tx}$ (lower=failing) |
|--------------------|---------------|----------------|----------------|----------------|
| stale_misleading | 8.2% | **0.0%** | 91.8% | −1.22 |
| stale_flaky | 6.1% | **0.0%** | 93.9% | −2.81 |
| (clean buggy) | 29% | – | – | −5.53 |
| (clean fixed) | 41% | – | – | +0.36 |

These argmax percentages should not be read as monotonic in the
`edit − noop` submargin: Qwen's first-token action prior over
`grep`/`edit`/`noop` can shift argmax rates even when the margin
and projection move in the expected pass/fail-transcript direction.

**The 0% explicit-`noop` rate under this prompt format.** Across 98
stale-evidence prompts the model never commits to explicit
abstention at first-token argmax; it defaults to `grep` >90% of
the time. The $u_{\rm tx}$ projection sits between the clean-buggy and
clean-fixed baselines; the residual direction tracks the
pass/fail-test evidence even when the emitted action does not
(§5.2 establishes that the residual direction is specifically
tracking transcript text, not a separately-encoded no-edit
decision).

**Two confounds bound the 0% number.** (a)
*Content-prior*, "the model specifically avoids the literal
token `noop`." Ruled out by App. G.7: swapping `noop` → `done`
or `skip` preserves the 0.0% abstention rate on all 499 Qwen prompts.
(b) *Position-prior*, "the model assigns lower probability to the
last-listed action." Addressed by the position-balanced and binary
controls in §5.5: on Qwen and CodeGemma, `noop` stays essentially
never-selected at every named-menu position, so the 0% canonical rate
is not primarily a last-position artifact for those two models.
DeepSeek differs under a single-token rerun, where the abstention label
is selected when listed first and nearly never otherwise; DeepSeek
action-level claims are therefore treated separately. These controls do
not turn first-token argmax into live-agent abstention behavior. A
thresholded projection can be used as a toy transcript-label flag in
this static prompt format, but App. G.8/G.10 show that direct transcript
parsing dominates it on clean transcripts; Appendix F contains the LOOCV
toy-monitor ROC at AUC = 1.000 in that setting.

### 5.1 Generalization to SWE-bench-Verified-derived paired prompts

We evaluate the **same frozen unit monitor direction** $u_{\rm tx}$
(legacy artifact name `v_noop`), computed once
on the toy tasks, never retrained, on **499/497/499 paired
prompts derived from SWE-bench Verified** (Qwen / CodeGemma /
DeepSeek; Qwen and DeepSeek use all 499 base pairs, while
CodeGemma uses 497 after two **2400-token-cap drops** to avoid
A10G OOM on the 7B model). The base set comes from the
500-instance Verified split with one instance dropped at
ingestion because its modified file was unavailable at the
base commit (file 404; full ingestion accounting in App. G.1).
Each prompt is constructed by extracting an 80-line oracle
window around each gold patch's largest Python hunk and
synthesizing a pytest transcript matching the buggy/fixed
condition. **This is not an end-to-end SWE-bench evaluation**: the agent is not given the repository, it does not propose
patches, and tests are not executed (the transcript text is
synthesized from FAIL_TO_PASS / PASS_TO_PASS metadata). It is a
static classifier evaluation on paired prompts that contain
test-pass/fail evidence, on the same prompt format used for the
toy substrate.

| metric | Qwen-1.5B | CodeGemma-7B | DeepSeek-1.3B |
|------------------------------|------------------------------|------------------------------|------------------------------|
| ROC-AUC [95% CI] | **0.989** [0.983, 0.994] | **0.950** [0.938, 0.961] | **0.888** [0.866, 0.907] |
| AP [95% CI] | 0.991 [0.986, 0.994] | 0.949 [0.935, 0.961] | 0.879 [0.851, 0.906] |
| precision (in-sample thr.) | 0.973 (468/481) | 0.894 (438/490) | 0.809 (414/512) |
| recall (in-sample thr.) | 0.938 (468/499) | 0.881 (438/497) | 0.830 (414/499) |
| fixed-condition FPR (in-sample thr.) | 0.026 (13/499) | 0.105 (52/497) | 0.196 (98/499) |

CodeGemma uses the all-49 toy contrast at (L26, pos −1) in all
headline monitor results; the 20-task responsive subset
is retained only for steering and exploratory SAE analyses
(App. G.4, H.6).

The positive class is the matched failing-transcript / buggy condition
(N = 499/497/499); the negative class is the matched passing-transcript /
fixed condition (same N). The fixed-condition FPR row counts
fixed/passing-transcript prompts whose classifier score
$s=-h\cdot u_{\rm tx}$ exceeds the threshold. It is a classifier
false-positive rate, not an observed committed edit action. AUC and AP CIs
are paired-bootstrap (B = 10000, seed = 0) over the full per-task
score vector (reproduction in App. J). **The bottom
three rows use the balanced-accuracy threshold chosen on each
model-specific evaluation set (499/497/499 for
Qwen/CodeGemma/DeepSeek)** and are therefore *in-sample*; they
are deployment-illustrative, not out-of-sample estimates, and we
report them in lower contrast than the threshold-free AUC/AP rows.
Held-out threshold checks for Qwen are documented in App. G.5
(50/50 split + leave-one-repo-out). The App. G.10 single-turn
veto simulation uses the Qwen held-out 50/50 mean threshold, a
computed DeepSeek held-out 50/50 mean threshold, and the
CodeGemma in-sample balanced-accuracy threshold under the all-49
direction (artifacts in App. J). **Formal held-out canonical-menu calibration
tables for CodeGemma and DeepSeek remain future work** (the raw §5.1 per-task
projection caches are not retained, so reproducing them needs model forwards);
App. G.17 does report a cross-model held-out operating point in the
disjoint-template paraphrase setting (threshold frozen on train templates).
App. G.3 documents per-repo calibration on CodeGemma in detail.

**The discriminative pass/fail-transcript signal is broadly
separable across layers; causal usage is localized by patching,
not AUC.** On the 499-prompt evaluation, projecting
$u_{\rm tx}$ at the *wrong* layer (L12) and same position gives AUC
**0.998**, *higher* than the L24 monitor's 0.989; the wrong-
*position* control (L24 / pos −8) drops substantially to 0.707; PCA1 at
(L24, pos −1) gives 0.977 (Adversarial table, App. G.2). Two
implications: (i) discriminative AUC by itself does not localize
the mechanism, the pass/fail-transcript evidence is linearly readable
along similar directions across many layers, and the choice of
(L24, pos −1) is justified by causal patching (§4.1), not by the
projection's discriminative power; (ii) the projection is
plausibly reading features that correlate strongly with explicit
pass/fail transcript text, a hypothesis we resolve in §5.2.

Each model uses a different argmax-action hedge, Qwen `grep`
90%/83% buggy/fixed, CodeGemma `edit` ≈99% in both, DeepSeek `view`
78%/85%, but the explicit-`noop` rate at first-token argmax is
**0%** across the 499/497/499 prompts (exact for Qwen and CodeGemma; a
first-subword proxy for DeepSeek, whose canonical `noop` is multi-token,
§\ref{sec:tok-audit}), while
the monitor's projection cleanly separates the conditions (gaps
+4.73 Qwen, +2.87 CodeGemma, +5.64 DeepSeek). On these static
prompts the residual direction tracks the pass/fail-transcript
evidence even though the emitted action does not pick `noop`;
note that the action distribution shifts (Qwen's `edit` rate
drops slightly on fixed prompts vs buggy, App. G.1) but a much
larger first-token prior favors `grep` / `edit` / `view` over
the abstention token. The evidence affects the `edit − noop`
submargin without overcoming this action prior. Whether a
*monitor-mediated agent* in an in-the-loop setting would convert
the submargin shift into fewer spurious edits at acceptable
recall is taken up by the agent-loop edit-veto in §5.9. Monitor-direction
baselines, the per-repo regression analysis driving CodeGemma's
10.5% in-sample fixed-condition FPR (concentrated in `django/django`
at 13.0% and `sphinx-doc/sphinx` at 22.7%), a per-repo calibration
counterfactual that drops the pooled in-sample fixed-condition FPR to
9.3%, and a sample-efficiency sweep showing that $u_{\rm tx}$ derived
from only 10 toy paired tasks is within 0.001 AUC of the full-49
baseline (§G.4) are in Appendix G.

![**Projection monitor on SWE-bench-Verified-derived paired prompts.** A frozen one-dot-product monitor at the Qwen causal site, or each other model's reported action-position cell, separates failing-transcript from passing-transcript prompts (AUC 0.989/0.950/0.888 for Qwen/CodeGemma/DeepSeek; full metrics in the §5.1 table). Threshold markers are in-sample operating points and are illustrative only. This is a static-prompt classifier evaluation, not an end-to-end SWE-bench agent run.](figures/main_monitor.png){#fig:monitor width=98%}

### 5.2 Contradictory-transcript control: what the projection actually reads

The §5.1 0.989 AUC has an obvious alternative explanation: the
synthesized pytest transcripts contain explicit lexical markers
(`FAILED test_x ... AssertionError` in buggy, `N passed in 0.04s`
in fixed; App. G.1). A direction that separates buggy from fixed
prompts might therefore be reading the **transcript text** rather
than the code's correctness. We resolve this with a 2 × 2
(code × transcript) contradictory-transcript control on Qwen, run
on the same 499 paired prompts, and run the same control on CodeGemma
and DeepSeek (cross-model table below).

**Design.** For each task we materialize four prompts (variant
`code_tests_swapped` in `src/no_op_circuit/dataset/schema.py`):

| code | transcript | label | source |
|--------|------------|--------------------|----------------------------------------------|
| buggy | failing | (B,B), matched | original §5.1 prompts |
| fixed | passing | (F,F), matched | original §5.1 prompts |
| buggy | passing | (B,F), swapped | new `buggy__code_tests_swapped.pt` |
| fixed | failing | (F,B), swapped | new `fixed__code_tests_swapped.pt` |

Both swapped cells reuse the *paired-task's own* opposite-condition
transcript verbatim, so transcript-text format is constant across
cells. Each prompt is scored with the **frozen** toy-trained
$u_{\rm tx}$ at L24/pos −1; score $s=-p$ per §5.1.

**Result.** Cell means and bootstrap (B = 10 000, seed = 0)
decomposition over the N = 499 paired tasks (Qwen-1.5B):

| code | transcript | mean score | interpretation |
|----------|------------|-----------:|-------------------------------|
| buggy | failing | **+2.18** | matched (B,B), high |
| fixed | failing | **+2.17** | swapped (F,B), high |
| buggy | passing | **−2.56** | swapped (B,F), low |
| fixed | passing | **−2.56** | matched (F,F), low |

| main effect | value | 95% CI |
|--------------------------|----------:|-----------------------|
| **ΔTranscript** (failing − passing, averaged over code) | **+4.733** | [+4.606, +4.862] |
| **ΔCode** (buggy − fixed, averaged over transcript) | **+0.002** | [−0.025, +0.029] |
| Interaction | +0.002 | [−0.010, +0.014] |

The transcript effect is large; the code effect is statistically
indistinguishable from zero (CI crosses 0 on N = 499 paired
tasks). As threshold-free metrics on the swapped off-diagonal
(998 prompts):

| labeling (positive class) | ROC-AUC | meaning |
|----------------------------|----------:|----------------------------------------------|
| **transcript = failing** | **0.988** | direction follows transcript text |
| **code = buggy** | **0.012** | anti-aligned with the code label: the direction tracks the (swapped) transcript, not the underlying code-semantics label (0.012 = 1 − 0.988, the same ordering scored against the inverted label) |

**Verdict.** On 499 SWE-bench-Verified-derived static paired
prompts, the $u_{\rm tx}$ direction tracks **pass/fail transcript
text**, not code semantics. The §5.1 AUC of 0.989 is essentially
the transcript-label AUC of 0.988 measured here; the +0.001 gap
is sampling noise. The §1 framing "carries the pass/fail-test
transcript evidence the model is given" should be read narrowly:
the *evidence* in question is the pytest transcript text, not
the underlying code correctness. Three implications:

- **The §5.1 monitor is a pass/fail-transcript-text classifier.**
 Anything an external system could compute by directly parsing
 the pytest transcript (e.g. a regex for `FAILED` / `passed`;
 App. G.8 reports such a baseline at AUC **1.000**) the
 projection monitor reads more indirectly. The mechanistic value
 of this paper is in *where* and *how* that evidence is read out at
 an action-token residual-stream site, not in offering a better
 classifier than a regex.
- **The mechanism story in §4 is still empirically valid but
 narrower in interpretation.** The causal patching peak at
 (L24, pos −1) on Qwen is the site at which substituting one
 prompt's residual into another's forward shifts the
 `edit − noop` margin. The §5.2 result clarifies that the
 *quantity being patched* is best understood as the residual
 carrier of pass/fail-test evidence at the action position,
 not a separately-encoded no-edit *decision*. App. G.9 bounds
 the no-transcript case for the tested Qwen L24 linear readouts:
 frozen $u_{\rm tx}$, a fresh contrastive direction, and a
 1536-D LR probe are all near chance (AUC ≤ 0.52); the toy
 `code` variant also shows no patching peak (App. D).
- **The SAE features (§5.4, App. H) re-interpret as pass/fail-transcript features.** The OMP-selected top features whose
 logit-lens promotions are `error` / `traceback` / `already` /
 `passed` are exactly the features one would expect a SAE trained
 on these prompts to learn for representing pass/fail
 transcripts. They should therefore be interpreted as pass/fail-transcript features rather than no-op-circuit elements.

**Cross-model reported-cell evidence (CodeGemma + DeepSeek).** We re-run the
identical 2 × 2 control on the other two models, scoring each at
its §4.3 reported patch cell with its own frozen $u_{\rm tx}$
(CodeGemma L26, all-49 direction; DeepSeek L22), all four cells
forwarded fresh server-side (App. G.16). The transcript-driven
verdict holds on all three (raw ΔTranscript magnitudes are in
each model's own projection scale and not directly comparable;
the model-invariant quantities are the |ΔCode|/|ΔTranscript| ratio
and the swapped-only AUCs):

| model (cell) | ΔTranscript [95% CI] | ΔCode [95% CI] | swapped-only AUC (transcript / code) |
|-------------------------|--------------------------|---------------------------|--------------------------------------|
| Qwen-1.5B (L24, N=499) | **+4.733** [4.61, 4.86] | +0.002 [−0.025, +0.029] | **0.988** / 0.012 |
| CodeGemma-7B (L26, N=496)| **+2.894** [2.80, 2.99] | −0.046 [−0.080, −0.013] | **0.951** / 0.049 |
| DeepSeek-1.3B (L22, N=499)| **+5.697** [5.46, 5.94]| −0.053 [−0.168, +0.061] | **0.888** / 0.112 |

On every model |ΔCode| ≤ 0.06 and |ΔCode|/|ΔTranscript| < 0.02; when
code and transcript **contradict** within a single prompt, the
projection follows the transcript (AUC 0.89–0.99) and is inverted
against the code label (AUC 0.01–0.11), because in the swapped
cells the transcript label is anti-correlated with the code
condition. ΔCode's CI crosses zero on Qwen and
DeepSeek (a genuine null); on CodeGemma it is a tiny but
statistically non-zero *negative* effect (−0.046, ~63× smaller
than ΔTranscript), if anything the projection very slightly
anti-tracks code, the opposite of a code-driven reading. The
keystone "reads transcript, not code" control is therefore
**three-model**, not Qwen-only.

We are careful not to over-read this in the other direction
either. The §4 causal patching site is real, the steering
intervention is real, and the projection is a meaningful
mechanistic observable for the pass/fail-transcript pathway. We
do **not** claim the projection is superior to direct transcript
parsing; its value here is mechanistic: it identifies where
explicit pass/fail evidence is read out at an action-token
residual-stream site and shows that this evidence affects the `edit − noop`
margin despite not dominating the first-token argmax.

Full analysis artifacts and scripts are listed in App. G.16.

### 5.3 Temporal separation: the transcript verdict survives upstream

§5.2 establishes that the projection reads the transcript text.
A natural objection is that this is then *trivial*: in every
result so far the transcript is the **last evidence block before
the `Action:` token**, so the direction may simply be reading a
`FAILED` token we inserted one block above the readout, exactly
what a `contains("FAILED")` regex does, and does better
(App. G.8, G.10). We test this by moving the transcript
**upstream**.

We reformat each of the 499 SWE-derived prompts (Qwen) into a
multi-turn agent trace: turn 1 shows issue + code; turn 2 is an
assistant `test` action and turn 3 shows the
**condition-matched pytest transcript** used in §5.1; then two **condition-neutral intervening turns**
(a `grep` with no
matches and a generic file view, byte-identical across buggy
and fixed and across all tasks) push the transcript upstream
before the final `Action:` decision. The decision-point's local
context therefore contains **no pass/fail tokens at all**. We
score with the *same frozen toy-trained unit monitor direction $u_{\rm tx}$* at (L24,
pos −1) used in §5.1, no retraining (full design + format
control in App. G.15).

| transcript position | projection ROC-AUC | turn-local regex | full-scrollback regex |
|---------------------|-------------------:|-----------------:|----------------------:|
| adjacent (§5.1) | 0.989 | 1.000 | 1.000 |
| **2 turns upstream (this §)** | **0.807** [0.788, 0.827] | **0.500** | 1.000 |
| absent, format-matched control | 0.509 [0.494, 0.525] | 0.500 | 0.500 |

Three things follow. **(i) The transcript verdict survives the
separation.** With the transcript two turns + the intervening
tokens upstream, the frozen direction still discriminates
buggy from fixed at **AUC 0.807**, degraded from the adjacent
0.989 (dilution is expected) but far above both the format
control (0.509, which holds everything constant except the
transcript and confirms the multi-turn *format* alone creates no
discriminability) and the §G.9 no-transcript floor. The
direction **carries the transcript verdict across intervening turns**, not a reading of an adjacent planted token. **(ii) It beats a
stateless turn-local regex (only).** A stateless turn-local
regex, one inspecting only the current observation before the
next action, reads chance (0.500), because the `FAILED` token
is two turns upstream. The residual carries the transcript verdict to the
decision point where a stateless local parser sees nothing.
This does **not** beat a full-history regex or a stateful parser
that records earlier test outcomes: both would recover the
upstream transcript (the full-scrollback regex is at 1.000
here). The value is mechanistic, a carried-forward within-context
representation, not deployment superiority over text parsing.
**(iii) The action hides what the residual reveals.** In this setting Qwen's first-token argmax is
*identical* across conditions (buggy 173 `edit` / 326 `grep`;
fixed 173 / 326), yet the projection separates them at 0.807, the clearest demonstration in the paper that the direction reads
evidence the emitted behavior does not surface, the same
represented-evidence-versus-action gap behind the 0% noop result (§5.1, §4.2).

This is a single forward pass over a constructed multi-turn
prompt, not a live agent loop with KV-cache continuation across
real turns; "temporally separated" here means positionally
distant within one context (the transcript is still textually
present, hence the full-scrollback regex still works). A
genuinely *evicted* transcript is the §G.9 no-transcript regime,
which is negative. Full design, the format-control variant, and
server-side scoring are in App. G.15.

### 5.4 SAE decomposition (exploratory): pass/fail-transcript features

Exploratory SAE decomposition is reported in Appendix H
(Fig. \ref{fig:sae}). The contrast direction is geometrically dense
in the learned SAE basis, and small OMP-selected feature subsets can
partially affect the `edit − noop` margin, but the feature identities
and behavioral effects are seed-fragile and artifact-specific (a
seed=0 re-run does not reproduce the exact magnitudes; App. H.7). In
light of §5.2 the top features are best read as pass/fail-transcript
features, not "no-op-circuit" elements. We therefore do not treat the
SAE decomposition as a core contribution.

### 5.5 Action-menu controls: near-zero abstention is position-robust under named menus

The 0% explicit-`noop` result (§4) could be a list-position artifact, since
`noop` is always last in the canonical menu. We re-score the 499 Qwen
SWE-derived `code_tests` paired prompts under three menu manipulations; all
five action names are single-token under the Qwen tokenizer (scripts in App. J).

**Action order (position-balanced).** Across five cyclic orderings that place
`noop` in each menu slot exactly once (4,990 prompts), `noop` is the argmax in
**1/4,990** prompts, ≤0.1% at every position, with per-position 95% bootstrap
CIs upper-bounded at 0.3% (Fig. \ref{fig:action_order}). Menu order does shift
the choice *among non-abstention actions* (when `noop` is first, `edit` 68% /
`grep` 32%; when `noop` is fourth, `grep` 100%) and modestly moves the
`edit − noop` margin (mean 0.95–2.40 logits by slot), but `noop` stays at mean
rank ≈4.7–5.0 of 5. The 0% result is therefore not primarily a position
artifact under the named menu.

**Binary `edit`/`noop`.** Removing `view`/`grep`/`test` does not unlock
abstention: `noop` is the argmax in 1.2% (buggy) / 2.8% (fixed) of prompts when
listed second and **0%** when listed first (1,996 prompts). So `grep`/`view`
are not merely investigation hedges concealing a latent preference to abstain;
the edit prior dominates even in a two-way choice. The small fixed > buggy gap
(2.8% vs 1.2%) is consistent with the transcript mattering at the margin.

**Abstract labels (A–E).** Presented as letters with an in-prompt mapping,
first-token choice is dominated by a *label-token* prior rather than action
meaning: Qwen emits `B` in 75% of prompts regardless of what `B` denotes
(it emits `B` on 89% of the prompts where `B` is not `noop`). The apparent 21%
"abstention" when `noop` maps to `B` is this `B`-prior, not a semantic decision
(abstention near 0% when `noop` maps to A, C, D, or E). A content-free baseline
confirms an intrinsic surface-form prior: with the same A–E letters carrying
**no** action meaning, Qwen picks `B` on **97%** of 998 prompts (the
`letter_only` control). Under abstract labels, first-token choice is therefore
governed largely by a letter prior, with action meaning only a minor modulation.

Together, the named-menu and binary controls show the near-zero `noop`
selection is not primarily caused by `noop` being last in the canonical menu
(position-robust on Qwen and CodeGemma, and surviving binarization). The
abstract-label control, however, reveals strong surface-form priors over
letters, so first-token selection remains heavily shaped by prompt-conditioned
token preferences. We therefore read this as a static prompt-conditioned
action-token preference, not live-agent abstention behavior, distinct from the
internal transcript-evidence representation of §4–§5.

**Cross-model.** On CodeGemma (action names single-token) the same result
holds: `noop` is argmax in ≤0.1% of prompts at every menu position
(2/4990) and 0% in the binary `{edit, noop}` menu, with `edit` chosen ~99%
throughout; its 0% is also position-robust. On DeepSeek, two action names
(`grep`, `noop`) are **not single-token** (§\ref{sec:tok-audit}), so we reran
the control with a single-token vocabulary `{view, find, test, edit, done}`
(all single-token). This **removes the multi-token confound for the
menu-order control and shows DeepSeek differs**: the
abstention action is the argmax **97.7% of the time when listed first** but ~0%
at every other position, so DeepSeek's near-zero abstention in the canonical
(abstention-last) menu is **largely a first-position bias**, unlike Qwen and
CodeGemma, which are position-robust. The position-robustness claim therefore
holds for Qwen and CodeGemma named menus, not DeepSeek.

| `done` (abstain) position | 0 (first) | 1 | 2 | 3 | 4 (last) |
|---|---:|---:|---:|---:|---:|
| argmax = `done` rate | 0.977 | 0.000 | 0.008 | 0.000 | 0.000 |

(DeepSeek single-token vocabulary `{view, find, test, edit, done}`,
N = 998 prompts per position, 499 SWE-derived `code_tests` tasks ×
2 buggy/fixed conditions; 4,990 rows total across the five cyclic
orderings, each placing `done` in one slot (artifact in App. J);
per-position argmax rate of the abstention action `done`.)

The abstract-label and content-free letter-prior controls, run on all three
models (A–E are single-token for each), confirm a model-specific single-token
**letter prior**, content-free, Qwen picks `B` 97%, CodeGemma `A` 78%, DeepSeek
`A` 56%. Under abstract labels this prior surfaces as spurious "abstention" when
`noop` happens to map to the preferred letter (Qwen 21% at `B`, DeepSeek 100% at
`A`), while CodeGemma is more meaning-sensitive (≤7%). So abstract-label
first-token selection is partly governed by a surface letter prior across all
three models, not action meaning. Artifacts and scripts are listed in App. J.

![**Action-menu controls (Qwen, 499 SWE-bench-Verified-derived `code_tests` prompts).** *Left:* `noop` argmax rate by its position in the action menu (0 = first); `noop` is selected in 1/4,990 prompts total, ≤0.1% at every position. *Right:* mean `edit − noop` logit by `noop` position and condition, order modulates the margin but never makes `noop` competitive. Binary-menu and abstract-label controls (text) tell the same story.](figures/action_order_control_qwen.png){#fig:action_order width=92%}

### 5.6 Five-action logit decomposition: the direction moves a submargin, not abstention

We report the `edit − noop` submargin, but Qwen's argmax is usually `grep` or
`edit`. To see what moving along the direction does to the *full* action
distribution, we decompose all five action logits under rank-1 additive
steering with $v_{\rm raw}$ (whose unit is $u_{\rm tx}$; the §4.2 dose-response
vector) at L24/pos −1 on the 49 toy `code_tests` tasks (reusing the §4.2 sweep;
scripts in App. J; Fig. \ref{fig:five_action}).

Steering with $v_{\rm raw}$ (unit $u_{\rm tx}$) moves the `edit − noop` margin
smoothly and monotonically (buggy mean 5.49 → 0.95 logits across α ∈ [−3, 3]) and
reallocates the argmax **between `grep` and `edit`**, `grep` falls from 98% to
4% (buggy) as α increases while `edit` takes over. But `noop` is **never** the
argmax at any steering strength (0% across all α and both conditions) and stays
at logit rank 5 of 5 throughout (clean `noop` sits ~2 logits below the next
action). So the direction modulates the edit-vs-abstain *submargin* and
dislodges the default `grep` hedge toward `edit`, but it does **not** induce
abstention or overturn the first-token action-token preference. Thus
`edit − noop` is a diagnostic submargin rather than a faithful two-action
policy model. This is the steering
analogue of the §5.4 SAE-ablation observation (flips are `grep → edit`, never
toward `noop`), shown directly on the causal direction. A **discrete-patching** five-action check on the same 49 toy tasks at
L24/pos −1 (paired residual substitution, not steering) confirms that `noop`
remains noncompetitive: `noop` is the argmax in **0/49** prompts under both F→B
and B→F patches. Discrete argmax changes occur within the `grep`/`edit` pair, F→B flips 13 of 49 buggy tasks from clean `edit` argmax to patched `grep`,
and B→F flips 12 of 49 fixed tasks from clean `grep` argmax to patched
`edit`. The direction of discrete `grep`/`edit` reallocations is
substrate-dependent (the §4.1 SWE-derived peak-cell check moves them the
opposite way, tracking the substrates' opposite clean `edit`-prevalence), so
we do not interpret these transitions as a stable full-action policy. The
robust five-action observation is submargin modulation with `noop`
non-induction.

**Five-action decomposition sanity check (Qwen + DeepSeek single-token).**
Table \ref{tab:five-action-cross-model} reports discrete five-action patching
at each model's reported action-position cell on the 49 toy `code_tests`
paired tasks: Qwen at the canonical-menu L24 causal readout site, and DeepSeek
rerendered with the single-token vocabulary `{view, find, test, edit, done}`
from §5.5 at its L22 reported cell (an exact single-token follow-up to the §5.5
menu-order control, *not* the canonical multi-token `noop`-proxy result of
§4.3). This is a *sanity check on what the intervention does to the full
action distribution*, not a causal-localization estimate: across both models,
under the current five-action renderer, the discrete intervention changes the
`edit − abstain` submargin in the expected direction while leaving abstention
noncompetitive, the abstention action (`noop` or `done`) stays the argmax in
**0/98 clean** (49 buggy + 49 fixed), **0/49** under F→B, and **0/49** under
B→F, with per-task argmax changes confined to non-abstention actions. Because the five-action
decomposition uses a newer renderer, we treat it as a sanity check rather than
as a replacement for the §4.1 causal-localization numbers; these
current-renderer shifts are not used to revise the archived §4.1/§4.3
causal-localization estimates, and artifact and reconciliation details are in
App. J.

\begin{table}[t]
\caption{\textbf{Five-action decomposition sanity check at reported
action-position cells (current renderer).} $N = 49$ toy \texttt{code\_tests}
paired tasks. Shifts are paired-bootstrap mean $[95\%~\mathrm{CI}]$ of the
signed change toward closure of the buggy/fixed margin gap in the
$\ell_{\rm edit} - \ell_{\rm abstain}$ margin (F$\to$B:
$m^{\rm clean}_{\rm buggy} - m^{\rm patched}_{\rm buggy}$, positive $=$ pushed
toward fixed; B$\to$F: $m^{\rm patched}_{\rm fixed} - m^{\rm clean}_{\rm fixed}$,
positive $=$ pushed toward buggy). \textbf{abstain} $=$ \texttt{noop} for Qwen,
\texttt{done} for DeepSeek's single-token follow-up. These current-renderer
shifts are a sanity check and are not used to revise the §4.1/§4.3
causal-localization estimates (App. J). The CodeGemma row is omitted pending an
unresolved renderer-comparability check (text).}
\label{tab:five-action-cross-model}
\centering
\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llrrll}
\toprule
model / menu & cell & F$\to$B shift [95\% CI], \% pos.\ & B$\to$F shift [95\% CI], \% pos.\ & abstain argmax (clean 0/98; F$\to$B; B$\to$F) & main argmax transitions \\
\midrule
Qwen-1.5B, canonical \texttt{view/grep/test/edit/noop} & L24/pos $-1$ & $+0.625\,[+0.546,+0.704]$, 96\% & $+0.596\,[+0.520,+0.672]$, 96\% & $0/98$ / $0/49$ / $0/49$ & F$\to$B: 13 \texttt{edit}$\to$\texttt{grep}; B$\to$F: 12 \texttt{grep}$\to$\texttt{edit} \\
DeepSeek-1.3B, single-token \texttt{view/find/test/edit/done} (\S{}5.5 follow-up) & L22/pos $-1$ & $+0.694\,[+0.500,+0.898]$, 67\% & $+0.714\,[+0.520,+0.908]$, 69\% & $0/98$ / $0/49$ / $0/49$ & F$\to$B: 7 \texttt{edit}$\to$\texttt{view}; B$\to$F: 9 \texttt{view}$\to$\texttt{edit} \\
\bottomrule
\end{tabular}}
\end{table}

**CodeGemma five-action decomposition (omitted, renderer comparability).** The
CodeGemma discrete five-action run at its L26 reported cell is omitted from
Table \ref{tab:five-action-cross-model} because its current-renderer numeric
margins are not comparable to the archived §4.3 cached artifact (a
renderer-comparability issue, not a challenge to the §4.3 reported-cell result).
Qualitative `noop` non-induction is unchanged, `noop` argmax remains **0/49**
under clean and patched conditions, with the numerical audit deltas and
artifact paths in App. J.

![**Five-action decomposition under steering along $u_{\rm tx}$ (Qwen, L24/pos −1, 49 toy `code_tests` tasks, buggy condition).** *Left:* argmax rate per action vs steering α, steering reallocates between `grep` and `edit`; `noop` stays at 0% throughout. *Right:* mean logit per action vs α, the `edit − noop` margin closes but `noop` remains lowest.](figures/five_action_decomp_qwen.png){#fig:five_action width=92%}

### 5.7 Transcript-robustness controls: the monitor is not robust-superior to text baselines

A standing concern is that the synthesized pytest transcripts are clean and
regex-solvable. We apply four noisy/degraded transcript transforms to the 499
Qwen `code_tests` prompts and compare the residual monitor (frozen `u_tx`)
against the App. G.8 regex classifiers and a bag-of-words logistic baseline
(5-fold CV). The monitor uses a lean final-position projection that reproduces
the headline AUC on clean `code_tests` (**0.989**), validating the scorer
(scripts in App. J).

| transcript | monitor | regex `contains` | regex n-FAILED | BoW |
|------------------------------|--------:|-----------------:|---------------:|------:|
| clean (`code_tests`) | 0.989 | 1.000 | 1.000 | 1.000 |
| + unrelated failure (flaky) | 0.728 | **0.500** | 1.000 | 1.000 |
| + many passing tests | 0.930 | 1.000 | 1.000 | 1.000 |
| truncated | 0.989 | 1.000 | 1.000 | 1.000 |
| summary line only | 0.896 | **0.500** | **0.500** | 1.000 |

(The line-density regex tracks n-FAILED here; it too is 0.500 on summary-only.)
Two readings. (i) The brittle presence regex `contains("FAILED")` breaks under a
single appended unrelated failure (0.50) and under summary-only transcripts
(0.50, the pytest summary says lowercase "failed"), where the count/density
regexes also break. (ii) But the monitor is **not** robust-superior: it degrades
under distractors (flaky 0.73, many-passing 0.93, summary-only 0.90) while a
**bag-of-words baseline holds AUC 1.00 on every variant**. In messy transcripts
the monitor beats the literal-token regex but is matched or beaten by
bag-of-words throughout, consistent with the monitor being an indirect
transcript-evidence readout, not a superior transcript classifier. A
deployment-relevant test would need transcripts where no surface text baseline
suffices. Caveat: the bag-of-words baseline is evaluated within each transform
distribution by 5-fold CV; it is a text baseline for the *same* distribution,
not a held-out format-generalization result. A separate
disjoint-vocabulary held-out-template paraphrase test (App. G.17) reverses this
comparison for the literal and train-vocabulary keyword baselines on Qwen,
where the frozen Qwen residual direction also beats a train-fit BoW baseline
(0.943 vs 0.750 pooled AUC). That test is template-based, and its held-out
robustness is Qwen-specific: CodeGemma and DeepSeek reported cells are negative,
and it is not evidence of a semantic abstraction. It therefore does not change
the present finding that bag-of-words matches or beats the monitor on these
noisy/degraded pytest-style transcript transforms. Artifacts and scripts are
listed in App. J.

### 5.8 Prior-gated, non-monotonic behavior across scale

The 1.5B 0%-abstention result (§5.5) is not a scaling law. Reading the first-token
argmax over the 499 SWE `code_tests` prompts at each size, the do-nothing (`noop`)
rate on passing (fixed) prompts is **non-monotonic**: 0 / 11.0 / 0 / 0 / 5.0% at
1.5B / 3B / 7B / 14B / 32B (95% bootstrap CIs: 3B $[8.4,13.8]$, 32B $[3.2,7.0]$;
7B and 14B are exactly 0). It is nonzero exactly for the two sizes whose dominant
first-token hedge is `edit` (3B, 32B) and zero when the hedge favors investigation
(`view`/`grep`: 1.5B, 7B, 14B). A position-balanced control at 7B confirms `noop`
stays near 0% at every menu slot, so this is not a last-position artifact; the
gate is which action the prior favors.

**Removing the hedge unmasks abstention (binary menu).** A binary {edit, noop}
menu (1,996 prompts/size) raises the passing-prompt `noop` rate for the view-prior
**7B from 0% to 11.9%** $[9.3,14.7]$ and for 3B from 11% to 42.8%
$[39.2,46.4]$, while 1.5B stays low (1.4%) and 14B/32B stay at 0%. So the gate is
the graded `edit`-vs-`noop` margin, which removing the investigation hedge
overcomes only when that margin is already near zero (7B, 3B), not for 14B/32B.

**Steering causally flips the action.** Rank-1 steering of the 3B evidence
direction ($v_{\rm raw}$ at L32) on passing toys drives the action from `edit` to
`noop`: the `noop` argmax rate rises 0→51% as $\alpha\to+6$, and the `edit − noop`
margin crosses zero at $\alpha\approx 4.0$. The signature that it rides the
*evidence* axis, rather than a generic abstention push, is that passing prompts
flip *before* failing prompts ($\alpha\approx+4$ vs $+6$). At the natural unit
($\alpha=1$) the action does not flip, consistent with 1.5B; the direction is
**action-capable but suppressed at the operating point by the prior**. Unlike the
refusal direction [@arditi2024refusal], which controls behavior, this evidence
direction flips the action only against a permissive prior or under amplification.

### 5.9 From static prompts to a live loop: over-editing and a deployable edit-veto

We move from the static single-turn decision to a minimal multi-turn agent loop to
ask whether the represented evidence predicts real over-editing and whether the
monitor is deployable as an edit-veto. **Protocol** (faithful decisions, simulated
execution): each turn the model picks one action from {view, grep, test, edit,
noop} given the running history; `view` reveals the oracle code window, `test`
reveals the condition's pass/fail transcript, `grep` is condition-neutral, and
`edit`/`noop` terminate. Observations are materialized from the paired data, so the
decisions are real model rollouts but no sandbox is executed. We run 40 SWE-derived
tasks in both conditions on 1.5B/3B/7B/14B/32B from two start states: *from-issue*
(issue + file list only, the agent must act to gather evidence) and
*evidence-present* (code + transcript in context). At each turn we record the
monitor projection ($h\cdot u_{\rm tx}$ at the size's causal cell); the **edit-veto**
blocks an `edit` whose projection reads "passing," using a **leave-one-out**
threshold.

**Over-editing is real and severe, and evidence in context does not stop it.**

| size | over-edit (from-issue) | over-edit (evidence) | + edit-veto (held-out) | recall (evidence) |
|---|---:|---:|---:|---:|
| 1.5B | 35% | 50% | 22% | 48% |
| 3B | 78% | 88% | **20%** | 90% |
| 7B | 78% | 98% | **27%** | 100% |
| 14B | 75% | 100% | 50% | 100% |
| 32B | 75% | 90% | **18%** | 100% |

For 3B–32B the agent edits already-passing code 75–100% of the time, and it does so
*more* with the passing transcript in context than without (14B edits 100% of
passing tasks given the transcript). This is the static "evidence represented but not
acted on" finding manifesting as real over-editing. 1.5B is the lone low point only
because it is *indecisive* (it exhausts the turn budget investigating, not
abstaining) and its recall on buggy tasks is poor (48%). A second bottleneck is
**evidence-gathering**: 3B runs `test` in only 2% of from-issue loops (it edits
blind), and for 7B testing *halves* over-editing (47% when it tests vs 100% when it
does not).

**The monitor is deployable as an edit-veto.** When the transcript is in context,
the internal pass/fail direction used as an edit-veto with a held-out threshold
cuts over-editing **2–5× at every scale** (88→20% at 3B; 98→27% at 7B; 100→50% at
14B; 90→18% at 32B), preserving 60–94% of correct edits (model-dependent; 3B
cleanest at 94% recall; decision-point AUC 0.68–0.89). The veto is useless without
gathered evidence (3B from-issue decision-point AUC 0.51, chance).

**A blunt prompt is brittle; the targeted veto is not.** An explicit "if the tests
pass you MUST `noop`" system instruction cuts 3B over-editing to 15% (recall 90%
preserved) but makes 7B `noop` *everything*, collapsing its recall to 8%; it
over-applies the rule. The evidence-conditioned veto cuts over-editing while
*preserving* recall, and in the realistic from-issue regime the instruction is moot
for models that do not gather the evidence (3B tests 2%). A targeted internal
signal beats a blunt instruction.

**Caveats.** Pilot scale (N=40), simulated execution (synthesized transcripts, no
real test runs), greedy menu-constrained action selection (not free-form
generation), and held-out (not cross-distribution) veto thresholds. The robust
signals are the *severity* of over-editing and its *multiplicative* reduction under
the veto; exact percentages will move with scale and protocol.

## 6. Discussion and Limitations

**The mechanism is the constant; the behavior is the variable.** Three results
across the Qwen2.5-Coder ladder (§4.4, §5.8–5.9) recast the static dissociation
detailed below as one corner of a graded picture. *(i)* The causal pass/fail
readout is **universal**, a late action-position site at every scale (1.5B–32B),
following a relative-depth law that drifts toward the final layer (width, not just
depth, pushes it deeper) and growing ~18× in causal effect. *(ii)* Whether that
evidence changes the **action** is **non-monotonic and prior-gated**: the
do-nothing rate on passing prompts is 0/11/0/0/5% (1.5B→32B), nonzero only when the
first-token prior is `edit`-dominant, and a binary menu or rank-1 steering can
override it; the evidence is action-capable but suppressed at the operating point.
*(iii)* In a live loop the represented evidence predicts **severe over-editing**
(75–100% of already-passing tasks for 3B–32B; unreduced by having the transcript in
context), which the internal direction, used as a held-out **edit-veto**, cuts 2–5×
at high recall. The original 0%-at-1.5B result is thus the small-model,
investigation-prior corner of this picture, and the monitor (mechanistic, not a
better transcript classifier) is nonetheless **deployable** when read internally
in the loop, where it beats a blunt "stop if passing" instruction that destroys
recall. The remainder of this section details the static foundation these results
build on.

**A represented-evidence-vs-action dissociation in coding-agent prompts.** Compute the classifier score
$s=-h\cdot u_{\rm tx}$ at `resid_pre[L_site, pos −1]` (Qwen causal peak / each other model's reported action-position cell) and threshold:
this one-dot-product projection separates matched failing-transcript/buggy
from matched passing-transcript/fixed prompts at
AUC 0.989 / 0.950 / 0.888 (Qwen / CodeGemma / DeepSeek;
499 / 497 / 499 paired prompts). The contradictory-transcript control
(§5.2; run on all three models in App. G.16 with
$|\Delta\mathrm{Code}|/|\Delta\mathrm{Transcript}| < 0.02$ everywhere)
shows the direction tracks **transcript text**, not code semantics.
**In the clean synthesized-transcript setting we evaluate, direct
parsing is the right implementation**, App. G.8 reports a
`contains("FAILED")` regex at AUC 1.000, and App. G.10's single-turn
edit-veto simulation has the regex (100% / 0% blocked / useful-edit
loss) strictly dominating the projection (84.4–100% / 11.9–29.2%
across three models). **Practical corollary: an internal-state edit-veto is no better than reading the transcript here.** The residual monitor is **not proposed as a
better transcript classifier**; its value is mechanistic, it
identifies where explicit pass/fail evidence is read out at the
action-token decision and shows this evidence causally moves the
`edit − noop` margin without dominating first-token argmax
(§4.1–§4.2; §4.1, Table \ref{tab:swe-peak-patching} makes the
decodability-vs-causal-use separation explicit: L12/pos −1 is more
decodable yet much smaller causally). The direction sits within the
representation-engineering [@zou2023representation; @rimsky2024caa]
line of read-and-control work, and 10 random toy pairs already reach
AUC within 0.001 of the full-49 baseline (App. G.4). The
no-transcript Qwen variant is **decisively negative**: at L24/pos −1
on the same 499 prompts, none of the tested linear readouts
discriminates buggy from fixed (frozen $u_{\rm tx}$ AUC 0.499; fresh
contrastive direction LOO 0.521; 1536-D LR probe LOO 0.518; App. G.9); the mechanism is bound to the transcript pathway for this
L24-linear-monitor approach.

**Temporal separation within context.** The multi-turn control
in §5.3 strengthens the mechanistic interpretation: the direction
is not only reading an adjacent pass/fail token, since moving the
transcript two turns upstream still leaves a 0.807 AUC signal at
the final action position. This result is Qwen-only and still
within a single rendered context; the transcript is positionally
distant but not evicted. It also beats only a stateless
turn-local parser, not a full-history or stateful parser that
records test outcomes as they appear. We therefore treat it as
evidence of a carried-forward within-context representation, not deployment
superiority over text parsing.

**Bounded Qwen paraphrase generalization.** App. G.12 runs each
model's paraphrase paired-prompt set (499/498/499 for
Qwen/CodeGemma/DeepSeek, the shorter paraphrase prompts incur
fewer CodeGemma token-cap drops than §5.1's 497) through one
deterministic 3-sentence natural-language paraphrase that
contains none of the regex's pytest tokens. On **Qwen** the projection at the canonical
(L24, pos=−1) gives AUC **0.995**: the readout is not the
literal `FAILED` token, which collapses to chance on the
paraphrase. This bounds the readout above the literal pytest
format but does **not** establish a semantic pass/fail
abstraction: the paraphrase is deterministic and keyed on
condition (buggy → "did not match" / "differed" / "does not
satisfy"; fixed → "matched" / "aligned" / "satisfies"), so any
keyword regex tuned to its vocabulary trivially achieves AUC ≈
1.0. The right interpretation is: *on Qwen, the direction
generalizes from pytest-format transcripts to one
paraphrase-format transcript*. A stricter **disjoint-vocabulary
held-out-template** follow-up on Qwen (App. G.17, no held-out fitting) addresses the keyword-leak: with
two train and two held-out templates whose discriminative content vocabulary
is disjoint, the frozen $u_{\rm tx}$ reaches pooled AUC **0.943** on held-out
templates while literal and train-vocabulary keyword baselines are exactly at
chance (a train-fit bag-of-words reaches pooled 0.750). This is still **not** a
semantic pass/fail abstraction: the templates are hand-authored and
deterministic, not adversarial or LLM-generated.

**Cross-model paraphrase status.** Replicating the held-out-template test on
CodeGemma and DeepSeek at their §4.3 reported cells (frozen directions, no
post-hoc search; App. G.17) shows the held-out robustness is **Qwen-specific**:
CodeGemma's direction degrades to held-out pooled AUC 0.649 and DeepSeek's
collapses to 0.619, both below the model-independent BoW (0.750). On the
deterministic App. G.12 paraphrase the two models likewise collapse at their
§4.3 reported cells (AUC 0.275–0.545). A post-hoc layer × position sweep
(App. G.13–G.14) identifies non-canonical cells with high
paraphrase AUC, CodeGemma at (L19, pos=−4) via the canonical
toy contrast (pytest 0.976, paraphrase-real 0.939), DeepSeek at
(L16, pos=−5) only after switching from a 49-pair toy contrast
to a 499-pair SWE-derived pytest-format contrast (paraphrase-real 0.925). We
report these as exploratory: both cells were selected on the
evaluation formats and lack held-out validation, and the
DeepSeek result is no longer the same methodology as the
canonical toy contrast. They support the weak claim that a
cross-format-discriminative cell *can be found under post-hoc
cell selection* on each model, not the stronger methodological
claim that the §4.3 reported-cell contrast would have found it. A multi-step transcript-observing agent harness on a
FixedBench-style stale/no-edit task suite using SWE-agent or OpenHands-style scaffolds and stronger paraphrase
baselines (keyword regex, BoW logistic regression, disjoint-vocabulary paraphrase templates, LLM-generated held-out
paraphrases, adversarial paraphrases) remain future work.

**The mechanism on Qwen, corresponding late-layer sites
elsewhere.** On Qwen a
single 1536-dim direction at (L24, pos −1) reproduces the entire
behavioral `edit − noop` gap as a continuous dose-response, with
bidirectional symmetry and a confirmed negative control on the
`code` (no-test-transcript) variant. The same reported-cell paired-patching
intervention applied to CodeGemma-7B (L26/28) and DeepSeek-Coder-1.3B
(L22/24) at relative depths in [0.857, 0.929] gives
action-position paired-patching effects at the reported cells on
each model (§4.3, App. D): F→B
+1.14 / B→F +1.18 for CodeGemma (on a bimodal substrate; the
§4.2 steering plot uses the 20-task responsive subset for a
cleaner dose-response, while the §5.1 monitor headline uses the
all-49 direction), F→B +0.19 / B→F +0.18
for DeepSeek (where the toy buggy/fixed gap itself is only +0.20,
so the *absolute* behavioral saliency is small even though the
reported-cell paired-patching effect is present). Across models, the causal evidence is uneven: Qwen has the full
bidirectional sampled grid and the max-statistic permutation null;
CodeGemma has an exploratory one-way F→B layer×position heatmap
(App. D) plus paired reported-cell values; DeepSeek has paired
reported-cell values only. We therefore treat the cross-model result as evidence for
corresponding late-layer action-position effects, not a fully validated
shared circuit. The fact that CodeGemma's SAE
behavioral specificity does not replicate Qwen's (App. H.6) is
consistent with the homology being looser than a strict shared
circuit.

**Toy-vs-SWE-derived saliency divergence on DeepSeek.** DeepSeek-1.3B's
toy buggy-fixed gap is +0.20 logits but the same frozen $u_{\rm tx}$
produces a +5.64-logit gap on the SWE-bench-Verified-derived eval
(monitor AUC 0.888), a 28× saliency divergence we surface but do
not disambiguate. The methodological implication is that toy
substrate alone is not a reliable predictor of cross-model monitor
strength (§6; App. G.4).

### 6.1 Limitations

This section summarizes the main limitations and the boundary of the claims. The
strongest claims in this paper apply to *static paired prompts
containing pass/fail test transcripts under a fixed five-action
vocabulary*; the bullets below mark the boundary between what
the evidence supports and what we treat as open.

**Characterized by controls in App. G / H:**

- *Transcript-text driver.* The §5.1 monitor follows transcript
 text, not code semantics (§5.2); a regex baseline matches its
 discrimination at AUC 1.000 (App. G.8). The Qwen lexical-redaction
 control (App. G.6) refines this: literal pass/fail tokens
 carry ~12 AUC points, broader transcript structure the rest.
 Under noisy/degraded transcripts (§5.7) the monitor is not
 robust-superior either: it degrades under distractors while a
 bag-of-words baseline holds AUC 1.00 on every variant.
- *0% noop rate is not a noop-token artifact.* In Qwen action-vocabulary
 swaps, `noop`→`done` and `noop`→`skip` both preserve 0.0% abstention rate (App. G.7).
- *Action-menu position effects characterized.* Position-balanced and
 binary named-menu controls show near-zero abstention is position-robust
 on Qwen and CodeGemma; abstract-label controls reveal strong
 surface-form letter priors. DeepSeek differs: the canonical `noop`
 label is multi-token, and a single-token rerun with
 `{view, find, test, edit, done}` shows a first-position bias. Therefore
 position-robust abstention claims are limited to Qwen and CodeGemma, and
 all first-token action claims remain prompt-conditioned rather than
 live-agent claims.
- *In-sample operating-point thresholds.* For Qwen, held-out
 50/50 + leave-one-repo-out give precision 0.96–0.97 (vs 0.973
 in-sample) and fixed-condition FPR 3.2–3.5% (vs 2.6%); App. G.5.
 DeepSeek's G.10 row uses a computed but not separately tabled
 held-out 50/50 threshold; CodeGemma's G.10 row remains
 in-sample under the all-49 direction. Formal CodeGemma /
 DeepSeek held-out calibration tables remain future work.
- *Single-turn edit-action veto characterized (not multi-turn).* The
 projection blocks 84.4–100% of fixed-condition first-token edit actions at
 11.9–29.2% useful-edit loss across three models (under the
 thresholds reported in App. G.10). A
 multi-step harness with retry-after-veto remains open (see
 below).
- *Temporal-separation control is Qwen-only and single-pass.* App.
 G.15 uses a constructed multi-turn prompt rendered as one
 context, not a live agent loop with KV-cache continuation or
 context eviction. The residual beats a stateless turn-local
 regex but not a full-history or stateful parser; the transcript
 remains textually present in the context.
- *SAE behavioral specificity is Qwen-specific and seed-fragile
 (exploratory).* The precise +34% margin reduction and 80%
 argmax-flip numbers do not survive a seed=0 re-train; the
 qualitative small-k subset effect (~30%) does (App. H.7).

**Scope limitations and open gaps:**

- *`grep`/`view` may be investigation, not a failure to abstain.* The
 non-`noop` argmax is usually `grep` or `view`, which are
 evidence-gathering moves; first-token selection cannot separate
 "investigating before deciding" from "declining to abstain."
 Distinguishing them requires a multi-step harness (future work item 1 below).
- *No code-only detection.* Without a transcript, the tested Qwen L24
 linear monitor is near chance (App. G.9); it is a transcript-evidence
 readout, not a code-correctness detector.
- *Not the agent-loop SWE-bench.* Our evaluation is on SWE-bench-
 Verified-*derived* static paired prompts (window + synthesized
 transcript), not the official agent benchmark
 [@jimenez2024swebench].
- *Toy substrate scope.* 49 LLM-generated single-file Python toys;
 untested on multi-file patches, longer contexts, or non-Python.
- *Three chat-fine-tuned code models.* Untested on base models,
 mixture-of-experts (MoE), or non-code instruction-tuned models. Monitor strength
 varies (AUC 0.989/0.950/0.888); DeepSeek's toy-vs-SWE-derived saliency
 divergence (§6) is unresolved.
- *Cross-model reported-cell follow-up is uneven.* §4.3 / App. D reported-cell
 values on CodeGemma (L26) and DeepSeek (L22) are not full
 bidirectional grids; the §4 CodeGemma steering plot uses a
 20-task responsive subset chosen post-hoc (the §5.1 headline
 uses all 49 toys).
- *In-context attribution is window-limited.* Cache covers only
 the last 32 token positions (App. I).
- *`edit − noop` is a submargin, not the full action decision.*
 Argmax rarely lands on `noop` regardless of projection. Qwen has
 steering-based and discrete-patching five-action decompositions
 (§5.6, §4.1); §5.6 adds a DeepSeek single-token `done` reported-cell
 discrete check, and the CodeGemma reported-cell discrete check is
 flagged as an unresolved consistency check (see §5.6 audit note).
 What remains open is full cross-model layer×position five-action
 localization, cross-model steering decompositions, and an exact
 DeepSeek treatment of the canonical multi-token `noop` menu via
 sequence-level action scoring.

**Still open future experiments,** in priority order:

1. **Real-execution agent-loop veto study.** §5.9 reports a *pilot*
 multi-turn loop with simulated observations (N=40, greedy
 menu-constrained actions) that already shows severe over-editing and a
 held-out edit-veto. What remains is a *real-execution* harness on a
 FixedBench-style stale/no-edit suite [@fixedbench2025; @yang2024sweagent]
 in which the agent runs actual tests, proposes free-form patches, and the
 veto includes retry-after-veto dynamics and per-model/per-repository
 threshold calibration at scale. The single most important follow-up.
2. **Full cross-model five-action *patching* decompositions.** Qwen has
 both steering-based and discrete-patching five-action decompositions
 (§5.6, §4.1); §5.6 adds a DeepSeek single-token `done` reported-cell
 discrete check (L22/pos −1) and reports a CodeGemma reported-cell
 discrete check (L26/pos −1) as an unresolved consistency check. What
 remains is *full bidirectional grids* on CodeGemma and DeepSeek
 (currently only paired reported-cell values), a CodeGemma rerun under a
 unified prompt renderer that reconciles with the §4.3 May-2026
 artifact, and a DeepSeek decomposition using sequence-level action
 scoring for the canonical multi-token `noop` label, or a fully
 consistent single-token abstention vocabulary used throughout the
 DeepSeek analyses.
3. **Held-out transcript-format robustness.** A disjoint-vocabulary
 held-out-template paraphrase test is now done on Qwen (App. G.17): frozen
 $u_{\rm tx}$ reaches AUC 0.943; literal and train-vocabulary keyword
 baselines are at chance; train-fit BoW reaches pooled AUC 0.750. The
 cross-model reported-cell follow-up is also done and is **negative**: at their §4.3
 reported cells (frozen directions, no post-hoc search) CodeGemma degrades to
 held-out AUC 0.649 and DeepSeek collapses to 0.619, so the robustness is
 Qwen-specific. What remains is adversarial and LLM-generated held-out
 paraphrases and train-on-one-format/test-on-another generalization.
4. **Full cross-model causal localization.** CodeGemma and DeepSeek
 bidirectional grids and max-statistic permutation nulls (currently:
 CodeGemma has an exploratory one-way F→B heatmap plus reported-cell paired
 values, App. D; DeepSeek is reported-cell paired only); DeepSeek should
 use a single-token abstention label (e.g. `done`) for exact
 action-level margins.
5. **Held-out threshold calibration for CodeGemma and DeepSeek** (the
 Qwen 50/50 + leave-one-repo-out procedure of App. G.5).

Beyond the §5.9 pilot (simulated observations, N=40), a real-execution
agent-loop study would need to define gate semantics, retry-after-veto
behavior, and per-model/per-repository threshold calibration at scale, with
actual test runs and free-form patches rather than the simulated
observations and menu-constrained actions used here; Appendix G.10 is a
separate single-turn edit-action veto simulation.

## 7. Conclusion

Our central finding is a represented-evidence-vs-action gap that is *graded across scale*: a model carries pass/fail evidence internally and it causally moves the edit-vs-abstain margin, but whether it changes the action depends on the first-token action prior. Across Qwen2.5-Coder 1.5B–32B the causal readout is universal (an action-position residual direction at relative depth 0.86–0.96, the effect growing ~18×), while the do-nothing rate on passing prompts is non-monotonic (0/11/0/0/5%) and gated by whether the prior is edit-dominant; a binary {edit, noop} menu and rank-1 steering can override the gate (§4.4, §5.8). Contradictory-transcript, no-transcript, and text-baseline controls bound what the static monitor is: it tracks transcript text, not code correctness, and a regex matches it when the transcript is present, so its static value is mechanistic rather than as a better classifier. In a live multi-turn loop, however, the represented evidence predicts severe over-editing (3B–32B edit already-passing code 75–100%, unreduced by having the transcript in context), and the same internal direction, used as a held-out edit-veto, cuts over-editing 2–5× at high recall and beats a blunt "stop if passing" instruction that destroys recall (§5.9). The mechanism is the constant and the behavior is the variable; the main open question is whether these gains survive a real-execution agent loop at scale (§6.1).

## 8. Ethics and Data Use

The 499 SWE-bench Verified instances we evaluate (App. G.1) derive
from public GitHub repositories under their respective open-source licenses
(`django` BSD, `sympy` BSD, `sphinx` BSD, `matplotlib` PSF-like,
`scikit-learn` BSD, `pylint-dev` GPL-2.0, `requests` Apache 2.0,
`pytest-dev` MIT, `pydata` BSD, `astropy` BSD). The Verified subset
is redistributed under SWE-bench's terms [@jimenez2024swebench];
we re-use 499 of its 500 instances (1 dropped at ingestion: file
404 at the base commit). We do not redistribute full upstream
source files or full oracle code windows. The activation-cache
release consists primarily of residual activations, projection
scores, run manifests, and limited prompt-derived metadata; as
documented in App. J.5, this metadata can include last-token text
or token ids that may contain short fragments of the oracle code
window.

The 49-task toy substrate (App. B.1) was generated by
`anthropic/claude-sonnet-4.5` (OpenRouter alias / likely
snapshot probe and prompt template in App. B.1.1; cf.
`scripts/generate_tasks.py`) and validated by
executing `pytest` in a sandbox: only `(buggy, fixed)` pairs where
`pytest` exits 1 on `buggy` and 0 on `fixed` were accepted (49 of
77 generated pairs; rejection log in `data/_generation_log.jsonl`).
Because this substrate is machine-generated research data, we
disclose its provenance and validation procedure explicitly; we
acknowledge it may inherit biases of the generator's data
distribution (e.g. coverage skew toward common Python idioms,
possible duplication of GitHub-derived patterns). The substrate is
released under the same license as the rest of the repository.

## 9. LLM Use Disclosure

The 49-task toy substrate was generated by
`anthropic/claude-sonnet-4.5` as described in App. B.1.1 and
validated by executing `pytest`. LLM coding-agent assistance was also
used to draft or modify analysis and paper-editing code, and for
editorial feedback, revision planning, and prose copy-editing; all
scripts were executed under author control, and all reported metrics
come from cached model outputs and analysis artifacts. LLMs were not
used to generate model logits or experimental measurements directly.
All reported numbers come from the scripts, cached activations, and
model outputs described in App. J.
The deterministic G.12 paraphrase strings and the G.17 train/held-out
paraphrase templates were hand-authored.

## 10. Broader Impacts

This paper's *mechanistic* finding is a residual-stream direction
at the action position that tracks pass/fail-test-transcript
evidence and causally modulates the `edit − noop` margin. App. G.9
rules out a no-transcript deployment for **this Qwen L24/pos −1
linear monitor specifically**; it does **not** rule out other
layers, nonlinear probes, full agent traces, or
semantic-code-analysis methods on the no-transcript distribution.
App. G.10 sketches a hypothetical edit-action veto: at the
action-token decision point of an agent that has been shown a
transcript, the projection wired as an `edit`-veto blocks
84.4–100% of fixed-condition first-token edit actions at
11.9–29.2% useful-edit loss across three models (under the
thresholds reported in App. G.10, Qwen and DeepSeek held-out
50/50; CodeGemma in-sample balanced-accuracy on the all-49
direction). We do **not** claim this is a deployment-ready gate; in the clean pytest setting the regex gate strictly dominates
(App. G.10), and we do not validate it in a multi-step agent
harness; that remains future work (§6.1).

The cost asymmetry between gate errors is deployment-specific.
In **interactive repair settings**, false-noops may be especially
costly because a real fix is silently suppressed by the gate
and the human reviewer never sees it; useful-edit loss is the
relevant column. In **high-volume stale-issue triage or
repository-cleanup settings** where most candidate tasks are
already resolved, spurious-edit reduction may dominate because
every unnecessary edit becomes review burden. We do not model
either cost regime end-to-end. In our single-turn setting the
regex gate strictly dominates the projection on both metrics
(App. G.10), so the cost-asymmetry calculus only becomes
interesting in settings where the trivial pytest-token regex gate is unavailable or insufficient
(held-out or adversarial paraphrases, genuinely ambiguous transcripts, or temporally-separated multi-turn traces when the comparison is restricted to stateless turn-local parsing).

**Why use a residual monitor at all?** On each model's
paired-prompt set (499/497/499 for Qwen/CodeGemma/DeepSeek)
the regex baseline beats the projection both as a classifier
(App. G.8: AUC 1.000 vs 0.989) **and** as an edit-action gate
(App. G.10: 100% / 0% vs 84.4–100% / 11.9–29.2%). In the clean
synthesized-transcript setting we evaluate, direct parsing is
the right implementation. One regime where the residual beats a
**stateless turn-local** regex is **temporal separation** (Qwen;
App. G.15): with the transcript moved two agent turns upstream,
the projection still discriminates at AUC 0.807 while a
turn-local regex, parsing only the current observation, reads
chance, because the decision-point context holds no pass/fail
token. A full-history regex or a stateful parser that records
earlier test outcomes would still recover it, so this is
mechanistic evidence that the model carries the upstream transcript verdict
to the action position, **not** deployment superiority over text
parsing. The remaining settings where a residual monitor might be
interesting are not the clean or deterministic formats tested here, but
harder settings where surface text baselines are deliberately held out
or fail: adversarial paraphrases, LLM-generated held-out paraphrases,
genuinely ambiguous transcripts, or live multi-step traces where the
relevant evidence is no longer turn-local. We test two Qwen paraphrase
settings: a deterministic paraphrase where keyword/BoW baselines also match the
monitor (App. G.12), and a disjoint-vocabulary held-out-template test where
Qwen reaches AUC 0.943 while literal and train-vocabulary keyword baselines are
at chance and train-fit BoW reaches pooled AUC 0.750 (App. G.17). Cross-model
reported-cell follow-up of the latter is negative (CodeGemma 0.649, DeepSeek
0.619), so this robustness is Qwen-specific. Neither paraphrase setting is
validated as a gate, and neither establishes a semantic abstraction.
The code-only absent-transcript setting is negative for this Qwen
L24 linear monitor (App. G.9). The mechanistic finding, identifying *where* explicit pass/fail evidence is read out at
an action-token residual-stream site and showing it causally affects
the `edit − noop` submargin, and that the direction **propagates
the transcript verdict across intervening turns** (App. G.15), does not
depend on the projection being a deployment-superior gate.

The same mechanism is **bidirectionally steerable** on Qwen
(§4.2): an adversary with model access could add $-\alpha\,v_{\rm raw}$
at the (L24, pos −1) cell to push the model toward `edit`,
amplifying over-editing, the opposite of the intended deployment
use. Detecting such tampering would require monitoring the
projection itself, not only the emitted action.

Generalization caveats compound this. Monitor strength varies
materially across model families: AUC ranges from 0.888
(DeepSeek) to 0.989 (Qwen), and in-sample fixed-condition FPR ranges
from 2.6% (Qwen) to 19.6% (DeepSeek). CodeGemma's fixed-condition
FPR also varies by repository: `django/django` is 13.0%,
`sphinx-doc/sphinx` is 22.7%, and the small `pylint-dev/pylint`
slice is 40.0% on only 10 pairs (App. G.3). Any future
deployment study would need per-model **and** per-(model,
repository) calibration; these integration considerations are
paper-internal, not a deployment recipe. **We
have not validated this monitor end-to-end in a real agent
loop**, and discourage deployments that skip that validation
step.

## References

\newpage
\appendix

# Appendix Contents {#appendix-contents .unnumbered}

**Contents.** Appendix A: extended related work. B: detailed methods.
C: behavioral Δ-margin and probe saturation. D: Qwen patching
heatmap, exploratory CodeGemma heatmap, and Qwen negative control. E: per-task steering dose-response
curves. F: stale-variant failure analysis and toy-monitor LOOCV.
G: full SWE-derived evaluation including specificity baselines,
per-repo CodeGemma regression analysis, and per-repo calibration
counterfactual; G also contains the temporal-separation control
(G.15), the cross-model contradictory-transcript control
(G.16), and disjoint-vocabulary held-out paraphrase robustness (G.17):
Qwen positive, CodeGemma/DeepSeek negative at their reported cells.
H: full SAE decomposition (Qwen and CodeGemma).
I: in-context attribution within the cached last-32-token window.
J: reproduction recipe and artifacts.

**Appendix evidence map.**

| Evidence location | Purpose | Status |
|----------|---------|--------|
| B–E | Qwen causal localization: toy substrate, hooks, patching, steering | full bidirectional sampled grid + max-statistic permutation null on Qwen; CodeGemma has an exploratory one-way F→B heatmap (App. D) plus reported-cell paired values; DeepSeek is reported-cell paired only |
| §4.1 SWE-derived peak-cell check | Qwen peak-cell paired patching on N=200 SWE-derived `code_tests` paired prompts at L24/pos −1 with L12/pos −1 and L24/pos −8 controls; all five action logits | supports transfer of the causal readout (F→B +0.314, B→F +0.311 at causal cell; ~10× smaller at L12, ~0 at pos −8; `noop` never argmax), not a full SWE-derived localization grid |
| §4.3 + §\ref{sec:tok-audit} | DeepSeek action-margin / patching analyses | first-subword proxy under canonical menu (`noop` multi-token); single-token `done` rerun in §5.5 covers menu-order control; §5.6 adds a single-token discrete patching follow-up at L22/pos −1 for the five-action decomposition |
| G.1–G.2 | SWE-derived ingestion and monitor baselines | supports monitor evaluation |
| G.8–G.10 | regex, no-transcript, single-turn gate controls | critical scope controls |
| G.12 | deterministic paraphrase control + keyword/BoW text baselines (run) | text baselines reach AUC 1.0; not a semantic proof |
| G.13–G.14 | cross-format post-hoc sweeps | exploratory, selection-biased |
| G.15 | temporal-separation control | Qwen-only mechanistic control |
| G.16 | cross-model contradictory-transcript control | critical transcript-vs-code control |
| G.17 | Disjoint-vocabulary held-out paraphrase robustness | Qwen frozen $u_{\rm tx}$ reaches AUC 0.943; CodeGemma and DeepSeek reported-cell directions degrade to 0.649/0.619; literal and train-vocabulary keyword baselines are at chance; train-fit BoW pooled AUC 0.750; template-based, not semantic abstraction |
| §5.5 | action-menu / position-balance controls (Qwen, CodeGemma, DeepSeek single-token rerun) | abstention position-robust on Qwen/CodeGemma; DeepSeek first-position bias |
| §5.6 | five-action steering + Qwen discrete-patching decomposition; DeepSeek single-token `done` reported-cell discrete check (L22/pos −1); CodeGemma reported-cell discrete check (L26/pos −1) flagged as unresolved consistency check vs. §4.3 (audit JSON) | Qwen and DeepSeek single-token follow-up move the `edit − abstain` submargin while abstention stays argmax 0/49 under all clean and patched conditions; CodeGemma submargin magnitudes not directly comparable to §4.3 pending rerun |
| §5.7 | noisy-transcript controls (monitor vs regex vs BoW) | text baselines match/beat the monitor |
| Methods / App. B tokenization audit | exact-context action tokenization | Qwen/CodeGemma single-token; DeepSeek `grep`/`noop` multi-token |
| H | SAE decomposition (figure in appendix) | exploratory, seed-fragile |
| I | in-context attribution | exploratory |
| J | reproduction and artifacts | reproducibility |

# Extended Related Work

The §2 references compressed several lines of evidence per cited
paper. The per-paper detail:

**Coding-agent action bias on stale tasks: TEBench and FixedBench.**
TEBench [@shang2026tebench] is a project-level *test-evolution*
benchmark that catalogs three subproblems, Test-Breaking (tests that need
modification because production code changed), Test-Stale (tests
that pass but no longer meaningfully validate updated behavior),
and Test-Missing, and asks coding agents to identify and update
tests. The Test-Stale subset is the piece closest to a "no-code-change" task, but the benchmark's framing is *test* evolution,
not no-code-change abstention. FixedBench [@fixedbench2025], from
the SRI Lab at ETH Zürich, is the more directly relevant
behavioral benchmark for this paper's phenomenon: it studies
*stale bug reports* where no code change is required and reports
undesirable code changes in 35–65% of cases, depending on
model/harness. Our paired-task substrate is methodologically downstream of
both; we treat the over-editing phenomenon as the behavioral
substrate to mechanistically localize, but we evaluate the
projection monitor only on SWE-bench-Verified-derived static paired
prompts that contain explicit pass/fail transcripts. FixedBench is
the behavioral phenomenon a future no-transcript no-edit detector
would target; App. G.9 is **negative** for the specific Qwen L24
linear monitor studied here (the tested linear readouts are near
chance without a transcript), so we do not claim a FixedBench-style
no-transcript detector (a gap we flag in §6.1).
The SWE-agent scaffold [@yang2024sweagent] and Agentless
decomposition [@xia2024agentless] are the dominant agent
architectures evaluated on SWE-bench; an agent-loop veto monitor
would sit at their localization-vs-validation seam.

**Refusal-as-a-single-direction.** @arditi2024refusal identify a
single residual-stream direction that mediates refusal in
instruction-tuned chat models; ablating it suppresses refusal,
adding it induces refusal. Our contrast direction is structurally
similar (one direction, contrastive mean-of-fixed minus
mean-of-buggy): the unit-normalized direction $u_{\rm tx}$ is
used as a read-out axis for monitoring, while the raw
mean-difference vector $v_{\rm raw}$ is used for additive steering
of the `edit − noop` margin in §4.2. But the *quantity it
controls* is narrower: it modulates a 2-action submargin under
code-plus-transcript prompts and tracks pass/fail-transcript
evidence (§5.2), not a separately-encoded abstention belief
analogous to refusal.
The methodological parent of the steering direction itself, mean-difference of paired prompt activations, is contrastive
activation addition [@rimsky2024caa]; we cite the older
*Activation Addition* [@turner2023actadd] for context, but the
specific paired-mean recipe used here is CAA.

**Activation steering for behavioral control.**
@turner2023actadd introduced *Activation Addition* (ActAdd),
demonstrating that a single additive direction in residual space,
computed from contrastive prompt pairs, can steer model behavior
without optimization or labeled data. @rimsky2024caa formalize the
*paired* mean-of-pairs variant (CAA) we actually use. Most directly
related, *Manifold Steering* [@huang2025manifold] shows that
overthinking in large reasoning models can be captured by a
low-dimensional manifold and steered by intervening along the
dominant direction. The umbrella *Representation Engineering*
[@zou2023representation] line frames the broader read/control
pipeline. Our monitor direction $u_{\rm tx}$ is the same kind of object
(mean-difference at a fixed (layer, position) cell), but identified
by causal localization on edit-vs-noop rather than chosen a priori
for overthinking behavior.

**Activation patching for causal localization.** Activation patching
[@meng2022rome; @wang2023ioi], and circuit-level analysis in
general [@elhage2021mathematical], are the standard tools for
identifying which residual stream sites carry information that a
downstream computation reads from. @heimersheim2024patching and
@zhang2024patchingbestpractices give practical guides and metric
recommendations (notably the logit-difference metric we adopt). Our
use of paired buggy/fixed prompts and the `edit − noop` margin
follows this protocol.

**Probes need control tasks.** @hewitt2019probes showed that a
high-AUC linear probe does not establish that the underlying model
*uses* the probed feature, high probe accuracy can reflect probe
capacity rather than feature usage. Our probe results saturate at AUC
≈ 1.0 across all layers; we treat them as necessary but not sufficient,
and rely on patching and steering for causal claims.

**Sparse features for code correctness.** @tahimic2025codecorrectness train
sparse autoencoders on code-LLM residuals and identify directions
correlating with code correctness, finding test-case attention is the
primary signal (echoing our finding that *test transcripts* are the
sole behavioral driver). They study correct-vs-incorrect *code
features*; we study the *agent's decision* to mutate the repository,
which is upstream of and distinct from correctness assessment, and we
localize to a single (layer, position) cell rather than a feature
collection.

# Detailed Methods

## Paired-task substrate

Each task `t` provides two task snapshots with identical issue text:
`buggy/parser.py` where `pytest` reports a failure, and `fixed/parser.py`
where it passes. We generated 49 such pairs across 12 archetypes via
`anthropic/claude-sonnet-4.5` and validated each by executing pytest in a
sandbox: only pairs where `fixed` exits 0 and `buggy` exits 1 are accepted.
We further evaluate five evidence variants per task:

- `issue_only`: bug report only.
- `code`: bug report + the buggy or fixed source.
- `code_tests`: + the corresponding pytest transcript.
- `stale_misleading`: bug report claims a bug, but the fixed code and
 passing transcript are shown (one-sided control).
- `stale_flaky`: as `stale_misleading` but with one synthetic unrelated
 flaky failure spliced in (e.g. `test_network_timeout` failing with a
 `ConnectionError`).

![Paired buggy/fixed substrate and intervention site.](figures/paired_task_diagram.png){width=90%}

### Toy task generation procedure

**Generator model.** All 49 toy pairs were synthesized by
`anthropic/claude-sonnet-4.5` (OpenRouter alias; cf.
`src/no_op_circuit/llm.py` `DEFAULT_MODEL`). The OpenRouter alias
resolves at call time to whatever Anthropic snapshot is currently
routed. The original generation run (2026-05-15) did not log the
provider's resolved `response.model` field, so the historical
snapshot used for the 49-task substrate is **irrecoverable** from
the existing `data/_generation_log.jsonl`. A re-probe of the alias
on 2026-05-17 routed to **`anthropic/claude-4.5-sonnet-20250929`**
(probe record at `data/_generation_snapshot_probe.json`); we treat
this as the most likely historical resolution but cannot confirm
it. The `llm.chat()` wrapper is patched to expose
`resolved_model` on `ChatResult`, and `scripts/generate_tasks.py`
writes it into `_generation_log.jsonl`'s `resolved_model` field, so
any future regeneration cannot lose this attribution.

**Sampling config.** Temperature `0.85`, `max_tokens=4096`, OpenAI
`response_format={"type": "json_object"}` for structured output,
no `top_p` override (provider default). Three retries with
4-second exponential backoff on `RateLimitError` /
`APIConnectionError`. Driver: `scripts/generate_tasks.py`
invoking `no_op_circuit.dataset.generator.generate_candidate`.

**Generation prompt.** The system message frames the task as
"propose a small Python paired (buggy, fixed) task for archetype
*X*"; the user message attaches one hand-curated few-shot example
(`parser_empty_input`) plus the target archetype's `description`,
`example_signature`, and `domains` (from `data/archetypes.yaml`).
The expected JSON schema specifies `task_id`, `issue_text`,
`primary_file`, `test_command`, `buggy_files`, `fixed_files`, and
a single `test_file`, full schema in `src/no_op_circuit/dataset/schema.py`.

**Acceptance gate.** Each candidate is materialized to a temp
directory and run twice: once against `buggy/` and once against
`fixed/`. Acceptance requires `pytest` exit code 1 on `buggy/`
**and** exit code 0 on `fixed/`. Additional gates: stdlib-only
imports (rejects external dependencies), no use of network
fixtures, and a single-file primary-file requirement. Rejections
are logged with reason at `data/_generation_log.jsonl`.

**Yield.** 77 attempts → 55 generation-stage accepts → 49 final
pairs after the additional file-level / validation gates (~37%
total rejection rate). Per-archetype generation-attempt counts:

| archetype | attempts | accepted (gen) |
|--------------------------|---------:|---------------:|
| off_by_one_loop | 14 | 10 |
| off_by_one_slice | 8 | 6 |
| missing_empty_case | 7 | 6 |
| missing_return | 4 | 4 |
| mutable_default_argument | 4 | 4 |
| case_sensitivity | 4 | 4 |
| wrong_arithmetic | 8 | 5 |
| wrong_comparison | 8 | 2 |
| wrong_default_arg | 4 | 3 |
| wrong_sort_order | 4 | 4 |
| swapped_args | 4 | 3 |
| shallow_copy_bug | 8 | 4 |
| **total** | **77** | **55** (49 final) |

The 49 final accepted tasks ship under `data/tasks/`; the full
rejection log is at `data/_generation_log.jsonl`.

## Action vocabulary

The prompt ends with `<chat-template-end>\nAction: ` and the model's first
predicted token is read out. We restrict the comparison to five named
actions, `view`, `grep`, `test`, `edit`, `noop`. The scalar of interest is
the `edit − noop` logit margin (single-tokenness audited below).

## Action-tokenization audit {#sec:tok-audit}

We verify single-tokenness in the **exact scored context**, the first token
after the final `Action: `, resolved by diffing `encode(prefix)` against
`encode(prefix+name)`, the same rule the action-logit readout uses
(`scripts/check_action_tokenization.py`; `results/tokenization/`). The five
names are single-token under Qwen and CodeGemma, but `grep` and `noop` are
**two tokens** under DeepSeek:

| model | action | scored form | token ids | single-token? | interpretation |
|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | all five | `view`/.../`noop` after `Action: ` | 1 id each (e.g. `noop` = 60829) | yes | exact action logits |
| CodeGemma-7B | all five | `view`/.../`noop` after `Action: ` | 1 id each (e.g. `noop` = 137734) | yes | exact action logits |
| DeepSeek-Coder-1.3B | view | `view` | 1820 | yes | exact |
| DeepSeek-Coder-1.3B | grep | `grep` | 70, 5520 | **no (2)** | first-subword proxy |
| DeepSeek-Coder-1.3B | test | `test` | 2806 | yes | exact |
| DeepSeek-Coder-1.3B | edit | `edit` | 10304 | yes | exact |
| DeepSeek-Coder-1.3B | noop | `noop` | 2459, 424 | **no (2)** | first-subword proxy |

Consequence: Qwen and CodeGemma action logits are exact single-token readouts;
for DeepSeek, `grep`/`noop` logits are first-subword proxies, so DeepSeek
named-action/action-menu logits on those names are first-subword proxies. We
therefore reran DeepSeek with the single-token vocabulary
`{view, find, test, edit, done}`
(`results/tokenization/deepseek_single_token_action_candidates.json`;
`results/action_order_control/deepseek_single_token_action_order_summary.json`).
The rerun (§5.5) shows a **first-position bias**, the abstention action is
argmax 97.7% when listed first and ~0% otherwise, so DeepSeek's near-zero
canonical abstention is largely positional, unlike Qwen and CodeGemma.

## Residual-stream hooks

We register PyTorch `forward_pre_hook`s on `model.model.layers[i]` of the
underlying HuggingFace causal-LM model (no use of TransformerLens; we work
directly on the underlying HuggingFace decoder stacks for Qwen2, Gemma,
and DeepSeek to avoid reimplemented-arch drift). Three context managers expose:

- `cache_forward(last_k=K)`: captures `resid_pre[L]`, `resid_post[L]`, and
 the final layernorm output at the last K positions per forward pass.
- `patched_forward([(layer, hook_point, position, value)])`: replaces the
 residual at the indicated (layer, position) with `value` for one forward.
- `steered_forward([(layer, hook_point, position, direction, alpha)])`:
 adds `alpha · direction` to the residual at the indicated cell.

All interventions act on `resid_pre` (the input to layer L), matching the
canonical activation-patching site in the mech-interp literature.

## Probe, patching, and steering analyses

1. **Probes.** Logistic regression at every (layer, position) predicting
 condition from `resid_post` features, with 5-fold stratified CV. Used as
 a necessary-but-not-sufficient check.
2. **Paired activation patching.** For grid experiments (Qwen, App. D),
 for each sampled (task, layer, position) cell, substitute the FIXED
 residual into the BUGGY forward (F→B) and measure the shift in
 `edit − noop` margin. **Bidirectional patching** additionally
 substitutes BUGGY into FIXED (B→F). For cross-model analyses
 (CodeGemma, DeepSeek), the analogous paired intervention is applied
 only at the reported patch cells (§4.3). The hypothesis-confirming
 sign is `clean_buggy − patched > 0` (F→B) and
 `patched − clean_fixed > 0` (B→F).
3. **Single-direction steering.** Compute the raw contrastive
 vector $v_{\rm raw} = \mu_{\rm fixed} - \mu_{\rm buggy}$ (§3) at
 the patching peak and add $\alpha\,v_{\rm raw}$ to the residual.
 Sweep `α ∈ {−3, −2, −1.5, −1, −0.5, 0, 0.5, 1, 1.5, 2, 3}` and
 record action logits per (task, condition, α).

## Models

Primary: **Qwen/Qwen2.5-Coder-1.5B-Instruct** (28 layers, hidden size 1536).
Cross-architecture models: **google/codegemma-7b-it** (28 layers,
hidden size 3072) and **deepseek-ai/deepseek-coder-1.3b-instruct** (24
layers, hidden size 2048). Gemma's chat template lacks a `system` role; we
fold system messages into the first user turn at render time. All forwards
run in `bfloat16` on Modal A10G or T4 GPUs.

# Behavioral delta-margin and probe saturation

The paired Δ-margin per variant for Qwen and CodeGemma (DeepSeek's
`code_tests` toy gap of +0.204 is reported in §4.3 instead of here;
the issue-only/code variants were not separately enumerated on
DeepSeek):

| variant | Qwen-1.5B (mean Δ, % positive) | CodeGemma-7B (mean Δ, % positive) |
|-------------|---------------------------------|-----------------------------------|
| issue_only | +0.000, 0/49 | +0.000, 0/49 |
| code | −0.001, 18/49 | −0.082, 9/49 |
| code_tests | **+0.659, 47/49 (96%)** | **+1.347, 29/49 (59%)** |

The `issue_only` and `code` variants produce no systematic action shift
in either Qwen or CodeGemma. Adding the test transcript drives a positive
shift toward `noop` in 96% of tasks on Qwen and a bimodal shift on
CodeGemma; DeepSeek's `code_tests` toy gap is small but positive
(§4.3). **Under these prompt variants, adding the test transcript is the only condition that produces a systematic behavioral shift.**

![Behavioral Δ-margin per task across three evidence levels and two models.](figures/behavioral_delta.png){width=90%}

**Probes saturate.** A logistic regression on `resid_post` predicting
`buggy` vs `fixed` from features at any (layer, position) achieves AUC ≈
1.0 across all 28 layers under `code_tests`, including layer 0 (raw
embeddings). This is uninformative on its own: the prompts contain very
different surface text (`FAILED` lines vs all-pass) and even the
embedding layer can trivially separate them. Probes confirm
that the information is widely available in the residual stream; they
do not identify *where the action computation uses it*.

# Qwen patching heatmap, exploratory CodeGemma heatmap, and Qwen negative control

Qwen F→B heatmap and exploratory CodeGemma F→B heatmap:

![Layer × position mean-shift heatmaps: Qwen F→B grid (left) and exploratory CodeGemma F→B grid (right). Only Qwen has the full bidirectional sampled grid and max-statistic permutation null; the CodeGemma panel is an exploratory one-way F→B sweep.](figures/patching_heatmap.png){width=92%}

**Specificity to test evidence.** On Qwen, the same bidirectional patching protocol
on the `code` variant (no test transcript) shows no positive peak. At the
canonical L24/pos −1 cell, F→B mean shift is **+0.015** (median +0.000,
45% positive, essentially chance), compared to +0.648 mean / +0.688 median
(100% positive) under `code_tests`. A small *negative* cluster appears at the very-late
layers L22–L26 / pos −1 (~−0.18 to −0.27), in the opposite direction of
the no-op shift: substituting unrelated residual content at this site
without test-evidence routing tends to weakly push the action toward
`edit`. We interpret this as the L24/pos −1 site being a
test-evidence-conditional readout; in the absence of an upstream
pytest transcript it does not carry the pass/fail-evidence
direction at all.

![Negative-control patching on the `code` variant (no test transcript).](figures/negative_control.png){width=60%}

**Multiple-testing correction on the grid maximum.** The Qwen patching
grid spans 14 sampled layers × 2 positions = 28 cells; selecting the
peak via `argmax` raises the obvious question of whether L24/pos −1
is the multiple-testing winner of sampling noise. We address this
with a max-statistic sign-flip permutation null. Under H_0 ("at every
cell the F→B per-task shift distribution is symmetric around zero"),
multiplying each task's shift vector by an independent ±1 sign is
exchangeable. For each of **B = 10000** permutations we redraw a
length-43 sign vector, recompute the per-cell mean shift, and
record the max across the 28 cells. The observed max (+0.6875 at
L24/pos −1) is **larger than every one of 10000 null maxima**
(null mean 0.055, null std 0.065, null 97.5th-percentile 0.228, null
max 0.446; **p < 1/10001** under the standard +1/+1 small-sample
correction). The max-statistic null distribution natively
accounts for the multiple-comparisons burden because each null draw
also takes the maximum across all 28 cells. Recipe and JSON record:
`scripts/compute_patching_grid_permutation.py` →
`results/patch_grid_permutation.json`.

# Steering dose-response per-task curves

Per-task dose-response across α ∈ {−3, …, +3}, Qwen + CodeGemma overlay:

![Single-direction steering dose-response curves; Qwen full N=49 (solid), CodeGemma responsive subset N=20 (dashed, right y-axis).](figures/steering_dose_response.png){width=90%}

This is a stronger claim than patching: patching replaces a 1536-dim
vector; steering shows the same `edit − noop` margin shift can be
reproduced by a single rank-1 additive intervention along the raw
contrastive vector $v_{\rm raw}$.
**A single labeled contrastive direction is sufficient to steer the
margin under our code-plus-transcript prompt distribution.** This
does not establish that pass/fail-transcript information is uniquely
one-dimensional or absent from higher-rank subspaces (PCA1 at the
same cell is comparable in §G.2, and probes saturate at every layer);
$u_{\rm tx}$ is the *labeled* direction that comes for free from paired-task mean differences and works directly under additive intervention.

# Stale-variant failure analysis and toy-monitor LOOCV

![Failure analysis under stale evidence (Qwen, N=49 per variant).](figures/failure_table.png){width=92%}

A monitor that thresholds the classifier score $s=-h\cdot u_{\rm tx}$ converts the
latent pass/fail-transcript signal into an overt transcript-based
fixed/passing label under this prompt format, not an
independent abstention/no-edit decision (the model itself does not
emit `noop` at first-token argmax). Under leave-one-out CV on the 49
clean buggy/fixed pairs the monitor achieves ROC-AUC = 1.000 and AP =
1.000, with 100% precision and 100% recall at the balanced-accuracy
operating point. A permutation null over 1000 label-shuffled LOOCV
runs (seed=0, `scripts/compute_auc_cis.py`) puts the observed AUC
of 1.000 at the 100th percentile of the null distribution
(null mean 0.501, max 0.660, p = 0.001), the LOOCV result is not a
small-N coincidence. We do not claim this clean separation persists
on stale-evidence prompts (where the latent projection compresses
toward zero); a thresholded toy projection uses the threshold learned
on clean pairs to flag prompts whose classifier score falls into the
fixed/passing region. Under this prompt format, that score can be
interpreted as a post-hoc transcript-based fixed/passing label even when
the model's own first-token argmax is not `noop`.

![Transcript-label projection monitor, ROC and precision-recall under leave-one-out CV (toy substrate).](figures/monitor_roc.png){width=92%}

# SWE-derived evaluation: full per-task tables and baselines

## SWE-derived ingestion and projection statistics

The SWE-bench-Verified-*derived* paired-prompt evaluation was conducted
on 499 / 497 / 499 instances (Qwen / CodeGemma / DeepSeek; 1 instance
dropped at ingestion because its modified file no longer exists at the
base commit; 2 additional CodeGemma instances dropped for exceeding a
2400-token cap needed to avoid A10G OOM on the 7B model). This is a
static-classifier evaluation, **not** the official SWE-bench agent
evaluation [@jimenez2024swebench] (no patch generation, no test
execution, no repository access). Each instance was ingested using a
deterministic script extracting an 80-line window around the gold
patch's largest Python hunk, applying the hunk in-memory to derive
the fixed counterpart, and synthesizing a pytest transcript from the
dataset's `FAIL_TO_PASS` and
`PASS_TO_PASS` lists (no Docker / pytest execution required).

Projecting `resid_pre` at each model's toy-trained intervention site
onto its unit monitor direction $u_{\rm tx}$ (projection
$p=h\cdot u_{\rm tx}$, lower = failing-transcript; gap =
mean-fixed − mean-buggy):

| condition / source | model | layer/pos | mean $p$ | fixed−buggy gap | N |
|---------------------------|---------------|--------------|-----------|-----------|----|
| SWE-derived buggy | Qwen-1.5B | L24 / pos −1 | **−2.18** | – | 499 |
| SWE-derived fixed | Qwen-1.5B | L24 / pos −1 | **+2.56** | **+4.73** | 499 |
| SWE-derived buggy | CodeGemma-7B | L26 / pos −1 | **−10.85**| – | 497 |
| SWE-derived fixed | CodeGemma-7B | L26 / pos −1 | **−7.98** | **+2.87** | 497 |
| SWE-derived buggy | DeepSeek-1.3B | L22 / pos −1 | **+13.16**| – | 499 |
| SWE-derived fixed | DeepSeek-1.3B | L22 / pos −1 | **+18.81**| **+5.64** | 499 |
| Toy clean-buggy reference | Qwen-1.5B | L24 / pos −1 | −5.53 | – | 49 |
| Toy clean-fixed reference | Qwen-1.5B | L24 / pos −1 | +0.36 | +5.89 | 49 |

On a 99-pair subset the two-model monitor scores AUC 0.993
(Qwen) / 0.958 (CodeGemma); on the full benchmark these regress
modestly (Δ −0.004 Qwen / −0.008 CodeGemma under the all-49
direction) to the §5.1 headline values as the broader
distribution surfaces more edge cases.

## Monitor-direction baselines: contrastive direction at the causal cell, signal everywhere else

Two original specificity controls on Qwen at the same site (L24,
pos −1, SWE-bench-Verified-derived N=99):

- **Random unit-direction baseline (N = 1000).** Drawing N = 1000 unit
 vectors uniformly on the 1535-sphere at (L24, pos −1) and computing
 the |AUC| each one attains as a signed-projection classifier yields a
 random-AUC distribution with mean 0.702, p95 0.916, p99 0.953, and
 max 0.983 (over all 1000 draws). **$u_{\rm tx}$'s 0.993 sits at the 100th
 percentile**: it beats every one of 1000 random directions. The
 random-direction max (0.983) is a stricter ceiling than the p99 and
 is still below $u_{\rm tx}$, so the gap is not a chance artifact.
- **Full-residual probe (1536 parameters, LOOCV).** A logistic-
 regression probe trained on the full 1536-D residual at (L24, pos
 −1) under leave-one-out CV over the 99 paired tasks reaches **ROC-
 AUC = 1.000** (likewise at L27/pos −1). $u_{\rm tx}$'s 1-D projection
 captures essentially all of the information the full-residual probe
 uses: 0.993 vs 1.000 (caveat: with d = 1536 ≫ n = 99 the LR probe
 has substantial in-LOOCV overfitting risk, so 1.000 is an
 upper bound on what's mechanistically separable; $u_{\rm tx}$'s 0.993
 with one degree of freedom is a much tighter claim).

These together imply that on the N=99 subset, **$u_{\rm tx}$, the
labeled direction the contrastive mean-difference recipe produces, is nearly as discriminative at the L24/pos −1 causal cell as
a full-residual probe at the same site, despite using no fitted
SWE-derived parameters**. We avoid stronger "information-equivalent
to a 1536-parameter probe" phrasing: the probe is high-dimensional
with in-LOOCV overfitting risk, the N=99 subset is small, and the
wrong-layer / transcript controls below weaken any "privileged
readout" framing. The adversarial baselines added below (wrong-layer same-position is comparably strong, wrong-position
collapses) clarify that the localization claim is the patching
peak (§4.1), and the readout-quality claim at that cell rather
than a unique discriminative axis in residual space.

**Adversarial baselines on the full N=499 SWE-bench-Verified-derived evaluation.**
The random-unit-vector baseline is intentionally weak (uniform vectors
on a 1535-sphere are near-orthogonal to any discriminative axis). We
add three harder controls on Qwen using the same paired-bootstrap
methodology as §5.1 (B=10000, seed=0; `scripts/adversarial_v_noop_baselines.py`):

| direction | site | AUC | 95% CI |
|----------------------------------------------|-----------------|---------:|-------------------|
| **$u_{\rm tx}$** (reference) | L24, pos −1 | **0.989**| [0.983, 0.994] |
| PCA1 of `resid_pre` | L24, pos −1 | 0.977 | [0.968, 0.984] |
| $u_{\rm tx}$ at WRONG layer | L12, pos −1 | 0.998 | [0.995, 1.000] |
| $u_{\rm tx}$ at WRONG position | L24, pos −8 | 0.707 | [0.674, 0.738] |

(Cross-model transfer, Qwen $u_{\rm tx}$ projected onto CodeGemma's L26
cell, is skipped: Qwen's 1536-D direction has no canonical mapping
to CodeGemma's 3072-D residual without a learned projection that
would entangle baseline-direction quality with projection-fit
quality.)

Three concrete reads of this table:

1. **At its own site, $u_{\rm tx}$'s edge over PCA1 is narrow** (0.989 vs
 0.977). The dominant-variance axis at (L24, pos −1) on the toy
 substrate is already nearly aligned with the buggy/fixed
 discriminator, so $u_{\rm tx}$ is not an arbitrary special
 construction; it is the labeled mean-difference direction that
 works directly without fitting. We do not over-claim that
 $u_{\rm tx}$ is qualitatively unique.
2. **Wrong-layer same-position is also strong** (L12/pos −1, AUC
 0.998). This is the same probe-saturation phenomenon noted in
 App. C: the buggy/fixed signal exists at every layer in
 `resid_pre`, but the §4.1 patching peak (L24) tells us where the
 action decision *reads it out* causally. The monitor's
 discriminative AUC is not, by itself, a localization claim, the
 activation patching is.
3. **Wrong-position is dramatically weaker** (L24/pos −8, AUC 0.707).
 Positions away from the action token at the same layer do not
 carry the discriminative signal at anywhere near the same
 strength: the action position itself is privileged.

## CodeGemma fixed-condition FPR: per-repository breakdown

(All numbers below use the all-49 CodeGemma monitor direction $u_{\rm tx}^{\rm CG}$ (artifact: `v_noop_cg`) at (L26, pos −1), the §5.1 headline direction; see `scripts/recompute_codegemma_all49.py`.)

The CodeGemma fixed-condition FPR at the operating point is 10.5%
(52/497, headline) vs Qwen's 2.6%. Per-task analysis on the 494
paired tasks in repos with N ≥ 5 attributes this gap to two
repos: **78% of the 51 fixed-condition false positives in this subset come from
`django/django` (30 cases, 13.0% repo rate) and
`sphinx-doc/sphinx` (10 cases, 22.7% rate)**, with
`pylint-dev/pylint` worst by rate (4/10 = 40.0%) on a small
sample. Qwen flags the same prompts correctly almost everywhere:
the misclassifications are near-threshold rather than
catastrophic and token-length is not the driver
(Mann-Whitney p = 0.20).

**Per-repo AUC and calibration counterfactual.** Per-repo AUCs
on the largest two repos are near the headline, the fixed-condition-FPR
concentration is a threshold-calibration issue, not a per-repo
discrimination collapse. Per-repo bootstrap 95% CIs (B=10000,
seed=0):

| repo | N_pairs | AUC | 95% CI | |
|------------------------------|--------:|---------:|-------------------|--------|
| django/django | 231 | 0.968 | [0.954, 0.981] | |
| sympy/sympy | 75 | 0.967 | [0.945, 0.986] | |
| sphinx-doc/sphinx | 44 | 0.975 | [0.948, 0.996] | |
| matplotlib/matplotlib | 34 | 0.983 | [0.958, 0.999] | |
| scikit-learn/scikit-learn | 32 | 0.951 | [0.909, 0.987] | |
| pydata/xarray | 22 | 0.983 | [0.955, 1.000] | † |
| astropy/astropy | 19 | 0.967 | [0.917, 1.000] | † |
| pytest-dev/pytest | 19 | 0.956 | [0.889, 1.000] | † |
| pylint-dev/pylint | 10 | **0.890**| **[0.780, 1.000]**| † |
| psf/requests | 8 | 1.000 | [1.000, 1.000] | † |

† marks repos with `N_pairs < 30`, where the CI is wide enough
that the per-repo AUC should not be treated as a reliable
held-out estimate.

Letting each repo (N_pairs ≥ 5) use its own balanced-accuracy
threshold instead of the global one drops the **pooled fixed-condition FPR from 10.3% to 9.3% at recall 87.9% → 91.9%** across
988 buggy/fixed observations, a 10% relative reduction in fixed-condition
false positives with a slight recall *gain* without retraining $u_{\rm tx}$. The
one per-repo outlier is `pylint-dev/pylint` (AUC 0.890 with a
wide CI on N_pairs = 10); future in-loop studies targeting a
pylint-heavy codebase should expect the bulk calibration recipe
to be less effective there and gather a small project-specific
calibration set. The reduction counterfactual is **in-sample**: each repo's
threshold is fit on its own data and evaluated on the same data;
a held-out (leave-one-task-out per repo) re-evaluation is on our
follow-up list.

## Sample efficiency of the monitor direction

A natural question: the headline monitor direction $u_{\rm tx}$
(legacy artifact name `v_noop`) is derived from
all 49 toy paired tasks, would 10 have sufficed? would 5? would
even 1? We sub-sample the toy substrate without replacement and
re-derive `v_noop_sub = mean(fixed[sample]) − mean(buggy[sample])`
at L24/pos −1 for each of $N \in \{1, 5, 10, 25, 49\}$, with 10
random subsamples per $N$ (deterministic single full-set draw at
$N=49$). Each subsampled direction is unit-normalized, projected
onto the **same 499 SWE-bench-Verified-derived paired prompts** the
§5.1 headline AUC 0.989 is reported on, and scored. **Caveat.**
Sample-efficiency is measured against the *same* evaluation set used
throughout the paper; once that set has been looked at, repeated
subsampling against it can become a form of implicit model selection.
A stronger sample-efficiency design would tune nothing on this set
and evaluate once on a held-out transcript-containing stale/no-edit
benchmark or a transcript-observing agent-loop harness; the result below is suggestive evidence of low
label cost, not a definitive efficiency claim. Computation:
`scripts/sample_efficiency_curve.py` →
`results/monitor_real/sample_efficiency.json`.

| N | mean AUC | std | min | max |
|----|---------:|-------:|-------:|-------:|
| 1 | **0.954**| 0.040 | 0.876 | 0.989 |
| 5 | **0.984**| 0.006 | 0.973 | 0.991 |
| 10 | **0.988**| 0.002 | 0.985 | 0.993 |
| 25 | **0.988**| 0.001 | 0.985 | 0.990 |
| 49 | **0.989**| – | – | 0.989 |

The curve **saturates by $N = 10$**: ten random paired tasks suffice
to produce a $u_{\rm tx}$ whose SWE-derived paired-prompt AUC is within 0.001 of the
full-49-task baseline. Even **$N = 1$**, a single paired-task
contrast, gives a mean AUC of 0.954 across 10 random single-task
draws, suggesting the direction is not solely an averaging artifact.
The variance
falls sharply: std drops from 0.040 at $N=1$ to 0.006 at $N=5$ to
0.002 at $N=10$.

![Sample-efficiency curve. AUC on the 499 SWE-bench-Verified-derived paired prompts (the same evaluation set used in §5.1), as a function of the number of paired buggy/fixed toy tasks used to derive $u_{\rm tx}$. Mean ± 1 std across 10 random subsamples per N; the N=49 full-set draw is deterministic. Saturation at N ≈ 10 (mean AUC 0.988 vs full-set 0.989). The evaluation set is the same throughout, so this is implicit-model-selection-vulnerable; a held-out transcript-containing stale/no-edit benchmark or a transcript-observing agent-loop harness would give a stronger reading.](figures/sample_efficiency.png){#fig:sample-efficiency width=85%}

Two reads of this. (i) **Low label cost for monitor bootstrapping
under our prompt format:** ~10 paired tasks suffice to reach within
0.001 AUC of the full-49 baseline on this evaluation set. (ii) The
contrast direction is detectable in the residual stream after even
a single paired comparison (mean AUC 0.954 at N = 1), so the
direction is not solely an averaging artifact. The result is suggestive of
low-label-cost monitor bootstrapping but is not itself a
deployment-readiness claim; see §6.1 for the controls a deployment
claim would require.

**Cross-model extension.** Re-running the same subsample sweep on
CodeGemma-7B (L26/pos −1, 497 SWE-derived prompts) and DeepSeek-
Coder-1.3B (L22/pos −1, 499 SWE-derived prompts) yields:

| N | Qwen mean AUC | CodeGemma mean AUC | DeepSeek mean AUC |
|----|--------------:|-------------------:|------------------:|
| 1 | 0.954 ± 0.040 | 0.825 ± 0.096 | 0.888 ± 0.045 |
| 5 | 0.984 ± 0.006 | 0.935 ± 0.024 | 0.897 ± 0.025 |
| 10 | 0.988 ± 0.002 | 0.935 ± 0.013 | 0.887 ± 0.013 |
| 25 | 0.988 ± 0.002 | 0.947 ± 0.009 | 0.887 ± 0.006 |
| 49 | **0.989** | **0.950** | **0.888** |

The same low-label-cost pattern appears on all three models under
this evaluation protocol: mean AUC at N=5 is within ~0.005, 0.015,
and 0.000 of the full-49 baseline on Qwen / CodeGemma / DeepSeek
respectively. DeepSeek's N=1 mean matches the full-set AUC (0.888),
though individual one-pair draws vary substantially (std ±0.045 at
N=1), consistent with the §6 toy-vs-SWE-derived saliency-divergence
observation.

**CodeGemma headline uses all 49 toys.** The CodeGemma N=49
number above (AUC **0.950**, AP 0.949, in-sample fixed-condition FPR 0.105) is
the §5.1 headline, computed from the all-49 direction; the
20-task responsive subset is retained only for §4.2 steering and
the exploratory CodeGemma SAE in App. H.6 (rationale in §5.1 and
§4.3).

\clearpage

![Cross-model sample-efficiency curves. AUC on each model's paired-prompt set (499/497/499 for Qwen/CodeGemma/DeepSeek) as a function of subsample size N, error bars are mean ± std across 10 random subsamples. All three models saturate by N ≈ 5–10. Compute: `scripts/sample_efficiency_curve.py`.](figures/sample_efficiency_cross_model.png){width=85%}

Outputs: `results/monitor_real/sample_efficiency.json` (Qwen),
`sample_efficiency_codegemma.json`,
`sample_efficiency_deepseek.json`.

## Qwen held-out threshold calibration

**This section reports Qwen-only threshold calibration; CodeGemma
and DeepSeek held-out calibration under the updated all-49
direction remains future work.**

The §5.1 operating-point precision/recall/fixed-condition-FPR values use a
balanced-accuracy threshold fit on the same model-specific
evaluation set (499/497/499 for Qwen/CodeGemma/DeepSeek), so
they are *in-sample*. Two held-out designs (no new
compute, just refitting the threshold on different splits of the
existing per-task scores; `scripts/held_out_threshold_calibration.py`):

| design | precision | recall | fixed-cond. FPR |
|-------------------------------------|--------------------:|--------------------:|--------------------:|
| **in-sample (§5.1, for reference)** | 0.973 | 0.938 | 0.026 |
| random 50/50 split, 200 seeds, mean | **0.964** | 0.938 | **0.035** |
| random 50/50 split, [2.5%, 97.5%] | [0.931, 0.983] | [0.908, 0.972] | [0.016, 0.072] |
| leave-one-repo-out (12 repos), pooled | **0.967** | 0.938 | **0.032** |

The held-out precision drops modestly (0.973 → 0.964–0.967), recall
is essentially unchanged (0.938 throughout), and the fixed-condition FPR
rises from 2.6% → 3.2–3.5%. The §5.1 numbers are slightly optimistic
but not catastrophically so; the monitor's *threshold-free* AUC of
0.989 is unaffected by this analysis, and the held-out operating
point remains at ≈ 96.5% precision with ≈ 94% recall on the
SWE-bench-Verified-derived paired prompts. Per-repo LOO threshold
distribution (`results/monitor_real/held_out_thresholds.json`)
shows that 9 of 12 repos give precision ≥ 0.95 on their held-out
fold; the outlier is `pylint_dev` (precision 0.692, N_fixed = 10),
consistent with the per-repo regression in App. G.3.

## Lexical-redaction controls (Qwen): literal tokens, redaction artifacts, and structural signal

§5.2 shows the §5.1 monitor follows transcript text. A natural
follow-up: how much of that is the *literal* `FAILED` / `passed`
tokens, vs a broader pass/fail representation that survives
lexical redaction? We run two levels of redaction.

**Level 1 (`code_tests_lex_redacted`).** Per-token mapping that
preserves character count within 1–3:

| original | replacement |
|-------------------------|-------------------|
| `FAILURES` | `REDACTED` |
| `FAILED` | `REDACT` |
| `AssertionError` | `RedactedErr___` |
| `passed` | `redact` |
| `failed` | `redact` |
| `Traceback` | `Trace________` |
| `OK\n` | `___\n` |

This redaction still leaks class
information: buggy transcripts get UPPERCASE replacements
(`REDACT`, `REDACTED`, `RedactedErr___`) while fixed transcripts
get the lowercase `redact`. Case and substring shape differ
between classes.

**Level 2 (`code_tests_lex_redacted_uniform`).** Stricter:
all pass/fail vocabulary maps to a single case-uniform token
`OUTCOME`. After this both buggy and fixed transcripts contain
literally `OUTCOME` tokens; only their *counts* and the
surrounding structural shape (failure-section length, summary
line) differ between classes.

We re-cache 499 paired SWE-bench-Verified-derived prompts on
Qwen-1.5B and score with the frozen toy-trained unit monitor direction $u_{\rm tx}$
(gap = mean-fixed − mean-buggy projection $p$):

| variant | ROC-AUC | mean gap |
|--------------------------------------|--------:|---------:|
| `code_tests` (§5.1) | **0.989** | +4.73 |
| `code_tests_lex_redacted` (Level 1) | 0.865 | +2.05 |
| `code_tests_lex_redacted_uniform` (Level 2) | **0.905** | +2.55 |

The uniform redaction (Level 2) actually gives a *slightly higher*
AUC than the per-token redaction (Level 1), 0.905 vs 0.865, so
the Level 1 case/substring artifacts were
not, in fact, doing the work. Both redactions converge on the
same picture: stripping the explicit pass/fail vocabulary costs
the projection ~8–12 AUC points, but the residual stays well
above chance. The direction is partly literal-token-driven and
partly structural (line count, presence of failure listing,
summary-line shape). We cannot, from these controls alone,
distinguish "the projection is responding to an abstract pass/fail
signal" from "the projection is responding to the structural
shape of pass/fail transcripts."
Appendix G.12 adds a deterministic paraphrase control, but stronger length-matched and held-out paraphrase controls would be needed
to disambiguate semantic readout from format-specific lexical/structural cues.

Outputs: `results/monitor_real/lex_redact_scores.json`,
`results/monitor_real/lex_redact_uniform_scores.json`.

## Action-vocabulary swap (Qwen): 0% explicit-noop is not a noop-token artifact

The 0% explicit-`noop` rate (§5) could in principle be an artifact
of the literal token `noop` rather than a fact about the model's
abstention prior. We re-cache the 499 paired prompts on Qwen-1.5B
with the action vocabulary replaced (`view`, `grep`, `test`, `edit`,
*synonym*) for two synonyms, and read off both the projection AUC
(transfer of $u_{\rm tx}$) and the synonym argmax rate.

| action vocabulary | ROC-AUC | projection gap | synonym argmax (buggy / fixed) |
|----------------------------------------------------|--------:|------:|--------------------------------|
| `view, grep, test, edit, noop` (§5.1, reference) | 0.989 | +4.73 | 0.0% / 0.0% |
| `view, grep, test, edit, done` | **0.990** | +4.65 | **0.0% / 0.0%** |
| `view, grep, test, edit, skip` | **0.988** | +4.55 | **0.0% / 0.0%** |

Two findings. (i) **$u_{\rm tx}$ transfers cleanly across action
vocabularies.** AUC and projection gap are essentially unchanged
when the abstention token is swapped, consistent with the residual
at the action position being computed before any action token is
generated. (ii) **The 0% explicit-abstention rate is not an
artifact of the literal `noop` token.** Replacing `noop` with
`done` or `skip` keeps the abstention argmax at 0% on both buggy
and fixed conditions across all 499 paired prompts; the model
defaults to `grep` >93% of the time under both alternative
vocabularies, with `edit` taking the rest. These swaps rule out the
narrow *content-prior* reading that the literal token `noop` is uniquely
avoided. The broader *position-prior* explanation is addressed by the
position-balanced, binary, and abstract-label controls in §5.5: on Qwen
and CodeGemma abstention stays near-zero at every named-menu position,
while a single-token DeepSeek rerun instead shows a first-position bias.

Outputs: `results/monitor_real/vocab_done_scores.json`,
`results/monitor_real/vocab_skip_scores.json`.

## The obvious baseline: a regex over the transcript text

Given §5.2's finding that the projection monitor is reading
transcript text, the natural baseline is a regex or pytest-summary
parser. We compute three trivial classifiers over the synthesized
transcripts (`scripts/regex_transcript_baseline.py`):

| classifier | ROC-AUC | AP | precision | recall | fixed-cond. FPR |
|--------------------------------|--------:|-------:|----------:|-------:|-----------:|
| `contains("FAILED")` | **1.000** | 1.000 | 1.000 | 1.000 | 0.000 |
| count of `FAILED` lines | **1.000** | 1.000 | 1.000 | 1.000 | 0.000 |
| failure-line density | **1.000** | 1.000 | 1.000 | 1.000 | 0.000 |

All three are perfect on each model's paired-prompt set
(499/497/499) because the synthesized buggy and fixed
transcripts differ deterministically (buggy contains FAILED
lines, fixed does not). The projection monitor's §5.1 AUC of
0.989/0.950/0.888 is **strictly worse** than this baseline.

**Implication.** The value of the residual-projection monitor
relative to a trivial transcript parser is not discriminative
power. It is mechanistic: it identifies the residual-stream
location and direction at which pass/fail evidence is read out
into the action-token computation and shows that this evidence
*causally* affects the `edit − noop` margin (§4.1, §4.2). If an
external system has the pytest transcript, it should parse the
transcript. The projection monitor is interesting only if one
of the following holds: (a) the model sees evidence the external
system cannot parse, (b) the transcript is paraphrased or noisy,
or (c) the monitor is needed at an internal decision point where
transcript evidence is temporally separated from the current
text (e.g.\ multi-turn agent traces). As App. G.15 shows, case
(c) only beats a stateless turn-local parser; a full-history
regex or stateful parser still recovers the upstream transcript.
App. G.9 reports the code-only no-transcript evaluation, decisively negative for the Qwen L24 linear monitor.

Output: `results/monitor_real/regex_transcript_baseline.json`.

## No-transcript SWE-derived evaluation (Qwen): bounding the L24-monitor approach

The single most important deployment-relevance question after §5.2
was: does the residual direction generalize to settings *without*
a transcript? If yes, the paper has a no-transcript stale-bug
deployment niche (FixedBench-style; @fixedbench2025). If no, the
mechanism is bound to the transcript pathway and the paper is a
mechanistic case study.

**Design.** We re-cache the 499 paired SWE-bench-Verified-derived
tasks on Qwen-1.5B under the existing `code` variant (issue text +
code, *no* pytest transcript; `src/no_op_circuit/dataset/schema.py`).
Both buggy and fixed conditions share the same prompt format,
differing only in the code itself. At Qwen L24/pos −1 we then run
three increasingly-strong tests on the resulting 998 residuals
(`scripts/no_transcript_full_analysis.py`):

| # | classifier | ROC-AUC |
|---|------------------------------------------------------|-------------:|
| 1 | **Frozen toy-trained $u_{\rm tx}$** (§5.1 transfer) | **0.4987** |
| 2 | Fresh `v_no_tx = mean(fixed_code) − mean(buggy_code)`, in-sample | 0.5432 |
| 2'| Same direction, paired LOO out-of-sample | 0.5213 |
| 3 | 1536-D LR probe at the same site, paired LOO | 0.5184 |

The fresh direction has $\| v_{\text{no\_tx}} \|$ = 0.156 (vs the
transcript-trained $\|v_{\rm raw}\|$ = 5.89) and cosine **−0.04** with
the frozen direction, *orthogonal*. The in-sample 0.543 is an
upper bound (the direction is fit on the same data it's evaluated
on); the paired-LOO out-of-sample drops to 0.521, and the LR probe
gives 0.518. **None of the tested linear readouts at L24/pos −1
discriminates buggy from fixed code in the absence of a
transcript** (frozen $u_{\rm tx}$, a fresh mean-difference
direction, and a 1536-D LR probe are all near chance).

The argmax distribution under no-transcript also matches the
"information-free" expectation: 80% `grep`, 20% `edit`, ~0%
`view` / `test` / `noop` on *both* conditions (the model can't
tell buggy from fixed apart and defaults to a `grep`-then-`edit`
hedge regardless).

**What this settles, narrowly.** Combined with §5.2 (contradictory-
transcript flips the score) and §C / App. D (toy `code` variant has
no patching peak), this establishes that:

- The mechanism this paper studies, *a linear residual direction
 at L24/pos −1 on Qwen*, is bound to the pass/fail-test-transcript pathway. There is no separate "no-edit decision"
 representation at this site readable along this direction.
- For this L24/pos −1 linear-residual-monitor approach, a
 no-transcript deployment is **not supported** by our data, we find no usable signal for the tested L24/pos −1 linear
 readouts to project onto. We
 do **not** claim this rules out *all* no-transcript detection:
 a different layer or position, a nonlinear probe, a full agent
 trace, or a semantic code embedding might do better. We also
 note this is not an actual FixedBench evaluation, only a
 no-transcript SWE-bench-derived ablation; we use the phrase
 "FixedBench-*style*" advisedly.
- The paper is a mechanistic case study of how a
 transcript-evidence representation at the action-token
 residual-stream site affects the action decision,
 **not** a stale-bug deployment paper.

Output: `results/monitor_real/no_transcript_full_analysis.json`;
cache: `results/`\allowbreak`cache-real-qwen-code-only-`\allowbreak`20260518T092253Z/`.

## Single-turn edit-action veto simulation: projection vs regex gate

§G.8's regex baseline classifies each model-specific paired-prompt
set (499/497/499 for Qwen/CodeGemma/DeepSeek) at AUC 1.000, so as
a classifier the projection is dominated. The
remaining defensible question is **as an edit-action gate**: when
the model's first-token argmax is `edit`, can the projection
veto fixed-condition (spurious) edits without killing too many
buggy-condition (useful) ones? We report this in the obvious
six-bucket form, alongside the same gate evaluation for the
regex baseline so the comparison is apples-to-apples.

**Important framing note.** This is **not** a multi-step agent
loop. The "agent" here emits a single first-token action; if it
is `edit`, the gate decides whether to allow or veto. A real
SWE-agent / OpenHands / Aider harness lets the agent `grep`,
`view`, run tools, iterate, and emit an actual patch, none of
which we simulate. This is a single-turn edit-action veto
simulation, not a multi-step agent-loop evaluation.

**Setup.** For each model's paired prompts (499/497/499 buggy
+ 499/497/499 fixed for Qwen / CodeGemma / DeepSeek), we record
the first-token argmax and apply two candidate gates:

- **Residual projection gate.** Project `resid_pre[L_site, pos −1]`
 onto each model's frozen unit monitor direction $u_{\rm tx}$
 (legacy `v_noop`) to get the classifier score
 $s=-h\cdot u_{\rm tx}$, where higher means
 failing-transcript/edit-warranted evidence. The gate vetoes an
 `edit` action when $s$ is below the model-specific threshold.
 Thresholds are model-specific: Qwen uses the App. G.5 held-out
 50/50 mean threshold; DeepSeek uses a computed held-out 50/50
 mean threshold stored in
 `results/monitor_real/agent_loop_simulation.json`; CodeGemma
 uses the §5.1 in-sample balanced-accuracy threshold under the
 all-49 direction. Formal held-out re-calibration for the
 all-49 CodeGemma direction remains future work.
- **Regex gate.** Allow the edit iff the prompt's pytest
 transcript contains the literal token `FAILED`.

Each prompt falls into one of six outcome buckets (A–F):

 A. useful edit committed (buggy, argmax=edit, ALLOW)
 B. useful edit not proposed (buggy, argmax!=edit)
 C. useful edit VETOED (buggy, argmax=edit, VETO) (bad)
 D. SPURIOUS edit committed (fixed, argmax=edit, ALLOW) (bad)
 E. spurious not proposed (fixed, argmax!=edit)
 F. SPURIOUS edit BLOCKED (fixed, argmax=edit, VETO) (good)

 spurious-edit reduction = F / (D + F)
 useful-edit loss = C / (A + C)

**Results.** Both gates evaluated on each model's paired-prompt
set (499/497/499 for Qwen/CodeGemma/DeepSeek):

| model | gate | spurious-edit reduction | useful-edit loss | final spurious-action rate |
|--------------------|------------------|------------------------:|-----------------:|---------------------------:|
| Qwen-1.5B | residual projection | 100.0% (85/85) | 29.2% (14/48) | 0.00% (0/499) |
| Qwen-1.5B | **regex over transcript** | **100.0%** (85/85) | **0.0%** (0/48) | **0.00%** (0/499) |
| CodeGemma-7B| residual projection | 89.4% (439/491) | 11.9% (59/496) | 10.46% (52/497) |
| CodeGemma-7B| **regex over transcript** | **100.0%** (491/491)| **0.0%** (0/496) | **0.00%** (0/497) |
| DeepSeek-1.3B| residual projection | 84.4% (65/77) | 23.9% (26/109) | 2.40% (12/499) |
| DeepSeek-1.3B| **regex over transcript** | **100.0%** (77/77) | **0.0%** (0/109) | **0.00%** (0/499) |

(Thresholds are the model-specific ones defined in the Setup
above. The CodeGemma row uses the all-49 direction, which gives
better gate metrics than the responsive-subset direction used
only for the steering plot, spurious-edit reduction 89.4% vs
84.7%, final spurious-action rate 10.46% vs 15.09%, at unchanged
useful-edit loss.)

**Reading.** In this clean synthesized-transcript setting, the
regex gate **strictly dominates** the residual projection on
every model: 100% spurious-edit reduction at 0% useful-edit loss.
This is unsurprising, the buggy and fixed transcripts differ
deterministically in containing `FAILED`, so a single-token
check is sufficient for the gate problem. The residual projection
is therefore **not** proposed as a deployment-superior gate in
this setting. What §G.10 does show is that the residual signal
is *aligned with* the right gate decisions, it correctly
identifies most fixed-condition first-token edit actions and
blocks them, but it is mechanistically informative, not
operationally superior to direct transcript parsing.

Settings where a residual gate would need separate evaluation
(none validated here) include:
(i) transcripts paraphrased into natural language;
(ii) transcript-text ambiguous (mixed pass/fail, flaky tests);
(iii) settings without `FAILED`-like keywords at all. On
temporal separation specifically: a constructed single-pass
temporal-separation setting is evaluated in App. G.15, the
residual beats a stateless turn-local regex but not a
full-history regex or stateful parser. What remains unvalidated
is a live multi-step agent loop with retry-after-veto dynamics,
KV-cache continuation, and possible context eviction. The
code-only absent-transcript setting is **decisively negative**
for this L24 linear monitor (App. G.9) and is **not** in the
future-niche list above.

**Limitations of this simulation,** beyond the regex dominance
above:

- **Single-turn.** A real multi-step agent that can `grep` after
 a veto and re-decide might pay smaller useful-edit costs;
 this remains future work.
- **First-token argmax, not patch generation.** The "agent" here
 produces a single action token; a real agent commits a multi-file patch with validation; "spurious edit" here means
 **fixed-condition first-token edit actions**, not committed
 PRs.
- **No retry / iteration cost modeled.** A veto in a real loop
 forces another action; that cost is not measured.

Outputs:
`results/monitor_real/agent_loop_simulation.json` (residual gate;
full per-(model, threshold) bucket counts);
`results/monitor_real/regex_gate_simulation.json` (regex gate);
`scripts/agent_loop_monitor_simulation.py`,
`scripts/regex_gate_baseline.py`.

## Robustness analyses: threshold sweep, per-repo, case study, layer sweep

Four post-hoc analyses on the existing cached scores (no new
compute) that bound the §G.10 gate claims and the
"broadly distributed signal" claim of §G.2.

**(a) Threshold-sweep Pareto frontier.** §G.10 reported two
operating points. The full sweep gives the gate trade-off
curve; the "knee" (max of spurious reduction − useful loss):

| model | knee thr | spurious reduction | useful-edit loss | final spurious |
|-------------|---------:|-------------------:|-----------------:|---------------:|
| Qwen | −0.878 | **95.3%** | **14.6%** | 0.80% |
| CodeGemma | −7.047 | 89.4% | 11.7% | 10.46% |
| DeepSeek | −18.265 | 74.0% | 8.3% | 4.01% |

For Qwen, the knee threshold trades 5pp of spurious reduction
(95% vs 100%) for 14.6pp of useful-edit recovery (15% vs 29% at
held-out), materially better than the balanced-accuracy default
in many plausible cost settings. CodeGemma's sweep is flat: the knee
threshold is essentially the in-sample balanced threshold
(−7.047 vs −7.054), so the §G.10 operating point is already
near-optimal. (CodeGemma row uses the all-49
`v_noop_cg` direction; the 20-task responsive-subset direction
gives knee values 85.7% / 12.5% / 14.08%.)

![Threshold-sweep gate trade-off curves. *(A)* Pareto frontier: spurious-edit reduction vs useful-edit loss across the score range, with knee points (filled circles). *(B)* Final spurious-edit rate vs useful-edit loss. Source: `scripts/threshold_sweep_deployment_curve.py`.](figures/threshold_sweep_deployment.png){width=98%}

**(b) Per-repo breakdown.** Across the 12 SWE-bench-Verified
repos at the held-out threshold (Qwen/DeepSeek) or in-sample
balanced threshold (CodeGemma, recomputed with all-49
`v_noop_cg`): **Qwen** achieves 100% spurious-edit reduction in
every repo where any spurious edits are proposed (final spurious
rate 0/499 pooled); useful-edit loss varies repo-to-repo (50%
sympy, 44% scikit-learn, 0% several others). **CodeGemma**'s
reduction now varies **60–100%** across repos with N≥5, with
`pylint-dev/pylint` the outlier (60% reduction at 40% final
spurious rate on N = 10; same outlier the §G.3 per-repo
regression flagged); `sphinx-doc/sphinx` sits next at 77% / 23%.
**DeepSeek**'s small-N repos show high-variance loss; large
repos (django, scikit-learn) sit at 80–100% reduction.

![Per-repo G.10 metrics for the three models, repos with ≥5 paired tasks. Colored bars: spurious-edit reduction. Gray bars: useful-edit loss. Source: `scripts/agent_loop_per_repo_and_casestudy.py`.](figures/per_repo_g10.png){width=99%}

**(c) Qwen over-veto case study.** The 14 Qwen Bucket-C cases
(useful edit proposed, monitor vetoed) reveal a clean failure
pattern: **13 of 14 buggy transcripts contain exactly 2 FAILED
lines and 5 total lines**; the remaining one has 4 FAILED lines
and 7 total lines. All 14 are also vetoed at the held-out
threshold. The pattern is mechanistically consistent with App.
G.6's structural-signal finding, SWE-bench tasks with only 2
failing tests produce the shortest buggy transcripts and sit at
the failure-end of the monitor's veto envelope. A practical mitigation for
a future in-loop evaluation would be to fall back to the regex
baseline (App. G.8) when the transcript has fewer than ~3
FAILED lines.

**(d) Layer-sweep AUC.** App. G.2 reported one wrong-layer cell
(L12, AUC 0.998); the full 28-layer curve at pos −1 generalizes
that finding. These curves are descriptive on the same SWE-derived
evaluation set used elsewhere, not held-out model-selection evidence: each
fresh `v_noop_L` is fit on the 49 toy paired tasks at layer L and then
evaluated on the 499 SWE-derived prompts, while the frozen L24 direction is
projected at different layers. The purpose is to show where the signal is
separable in residual space, not to claim held-out generalization:

| layer | fresh `v_noop_L` AUC | frozen L24 AUC |
|:-----:|---------------------:|---------------:|
| 0 | 0.500 | 0.500 |
| 5 | 0.999 | 0.732 |
| 15 | 0.993 | 0.982 |
| 20 | **1.000** | 1.000 |
| 24 | 0.989 | 0.989 |
| 27 | 0.985 | 0.983 |

A fresh per-layer contrastive direction reaches AUC ≥ 0.99 from
L5 onwards and peaks at **L20**, not L24. The frozen L24-direction
transfers cleanly from L15+ and is inverted at very-early layers
(e.g. L3 AUC 0.27, expected, early-layer residuals are
token-feature-dominated). Implication: the §4.1 causal patching
peak at L24 is a causal *readout* site for the `edit − noop`
submargin under intervention, not where the pass/fail signal is
most separable, separability and causal usage are distinct
questions, and the discriminative peak (L20) is upstream of the
causal site by 4 layers.

![Layer-sweep AUC at pos −1 on Qwen (28 layers, 499 paired prompts). Fresh `v_noop_L` (per-layer contrastive direction, fit on the 49 toy tasks at layer L) vs frozen L24-trained `v_noop` projected at each layer. Red dashed line marks L24. Both curves are descriptive on the same SWE-derived evaluation set used elsewhere (not held-out model selection). Source: `scripts/layer_sweep_auc.py`.](figures/layer_sweep_auc.png){width=92%}

Outputs:
`results/monitor_real/threshold_sweep_deployment.json`,
`agent_loop_per_repo_{qwen,codegemma,deepseek}.json`,
`agent_loop_casestudy_qwen.json`,
`layer_sweep_auc.json`.

\clearpage

## Paraphrased-transcript control: Qwen's direction generalizes to one NL paraphrase format; CodeGemma and DeepSeek collapse at the reported cell

A natural question is whether the
projection reads the *literal* pytest format (`FAILED`,
`AssertionError`, summary lines) or a more *semantic* pass/fail
signal that would survive a format change. App. G.6's lexical
redaction is a partial answer (AUC dropped 0.989 → 0.865 → 0.905
under increasingly strict redaction, leaving the question
"what was the residual 0.9 AUC reading, semantic pass/fail or
transcript structure?"). This appendix runs the cleaner test:
**replace the synthesized pytest transcript with a fixed
natural-language paraphrase that contains no pytest vocabulary
and is parallel in structure across buggy and fixed.**

**Design.** Two paraphrase variants, both deterministic and
keyed only on the buggy/fixed condition (`code_tests_paraphrased_minimal`
and `code_tests_paraphrased_realistic` in
`src/no_op_circuit/dataset/schema.py`):

| variant | buggy text | fixed text |
|-----------|----------------------------------------------------------------------------|--------------------------------------------------------------------------|
| minimal | "The expected behavior was absent in the output." (49 chars) | "The expected behavior was present in the output." (50 chars) |
| realistic | 3-sentence prose, ~280 chars, parallel structure: "the assertions did not match...output that differed...does not yet satisfy the specification" | 3-sentence prose, ~270 chars, parallel structure: "the assertions matched...output that aligned...satisfies the specification" |

Neither variant contains `FAILED`, `failed`, `passed`, `OK`,
`AssertionError`, or `Traceback` (verified). The prompt
structure is preserved, the code-block wrapper and `### Test output`
header are unchanged, so only the transcript-block content changes
relative to §5.1.

We re-cache each model's paired-prompt set under each variant on
**all three models** (Qwen-1.5B, CodeGemma-7B,
DeepSeek-Coder-1.3B; the shorter paraphrase prompts incur fewer
token-cap drops than §5.1, so the paraphrase paired-N is
499/498/499 for Qwen/CodeGemma/DeepSeek) and score with each
model's frozen toy-trained unit monitor direction $u_{\rm tx}$ at its §4.3 reported patch cell
(L24 / L26 / L22 respectively). Projection columns report
$p=h\cdot u_{\rm tx}$ (lower = failing); gap = mean-fixed − mean-buggy.

**Cross-model result.**

| model | format | projection AUC | mean-buggy $p$ | mean-fixed $p$ | fixed−buggy gap |
|------------------------|-------------------|---------------:|----------------:|----------------:|------------:|
| Qwen-1.5B (L24/-1) | pytest (§5.1) | **0.989** | (lower) | (higher) | +4.73 |
| Qwen-1.5B | paraphrase minimal | **0.753** | +0.56 | +1.85 | +1.28 |
| Qwen-1.5B | paraphrase realistic| **0.995** | −1.56 | +3.02 | +4.58 |
| CodeGemma-7B (L26/-1) | pytest (§5.1) | **0.950** | – | – | +2.87 |
| CodeGemma-7B | paraphrase minimal | 0.509 (chance)| −6.92 | −6.85 | +0.07 |
| CodeGemma-7B | paraphrase realistic| **0.275** | −8.13 | −9.25 | −1.13 |
| DeepSeek-1.3B (L22/-1) | pytest (§5.1) | **0.888** | – | – | +5.64 |
| DeepSeek-1.3B | paraphrase minimal | 0.475 (chance)| +11.60 | +11.24 | −0.36 |
| DeepSeek-1.3B | paraphrase realistic| 0.545 (chance)| +12.69 | +13.44 | +0.75 |

All AUCs in this paraphrase table use the model-specific paired-Ns
(499/498/499 for Qwen/CodeGemma/DeepSeek; the shorter paraphrase
prompts incur fewer CodeGemma token-cap drops than §5.1's 497)
unless otherwise noted.

**At the §4.3 reported patch cells (the Qwen patching peak and the
CodeGemma/DeepSeek reported cells), the paraphrase test generalizes on
Qwen and collapses on the other two models.** On
Qwen, the projection's AUC under the realistic NL paraphrase is
**0.995**, essentially identical to the §5.1 pytest baseline of
0.989, and 0.753 under the minimal paraphrase. On CodeGemma and
DeepSeek, both paraphrase variants give AUC at or below chance
(0.509, 0.475 minimal; 0.275, 0.545 realistic). CodeGemma's
realistic-paraphrase AUC of 0.275 is *inverted*, the mean
projection is *more negative* on fixed than on buggy, reversing
the §5.1 sign convention.

**What the Qwen 0.995 does and does not establish.** The
`FAILED`-regex collapse on the paraphrase (AUC 0.5) rules out one
specific surface cue: the projection is **not** reading the
literal pytest token. But the paraphrase is *deterministic and
keyed on condition*, every buggy prompt receives the same
sentence ("the assertions did not match … output that differed …
does not yet satisfy the specification"), every fixed prompt the
same parallel sentence ("the assertions matched … output that
aligned … satisfies the specification"), so a keyword regex
tuned to the paraphrase's vocabulary (e.g. `"did not"`,
`"differed"`, `"does not yet satisfy"`) trivially achieves
AUC ≈1.0 by construction. We confirm this directly: a bag-of-words
classifier and a single generic failure/pass keyword lexicon (not tuned
to either style) both reach **AUC 1.0** on the minimal *and* realistic
paraphrases (`scripts/paraphrase_baselines.py`;
`results/paraphrase_baselines/`), matching the residual monitor's 0.995
and far above the literal-`FAILED` regex's 0.5. We do **not** disambiguate between
(a) the projection is responding to a more abstract pass/fail
signal that survives format change, and (b) the
projection reads paraphrase-format lexical cues. The defensible
claim is **the projection generalizes from one transcript format
(pytest) to one paraphrase-format transcript on Qwen**; that is
above the literal pytest token, but below "semantic pass/fail."

**Text baselines (now run) and remaining paraphrase controls.** A keyword-regex over the paraphrase's discriminating
tokens (`"did not"` / `"matched"` / `"differed"` / `"aligned"`
/ `"satisfies"` / `"does not yet satisfy"`) is one such control;
bag-of-words logistic regression over the paraphrase block is
another; negation-count and length-matched paraphrases are
others; multiple paraphrase templates with disjoint discriminative
vocabulary would test cross-template robustness; an LLM-generated
held-out paraphrase suite (varying surface form while preserving
pass/fail meaning) would test generalization beyond two manually
authored formats; an adversarial paraphrase where pass/fail
meaning is preserved but the surface polarity words are swapped
or removed would directly test semantic abstraction. The keyword-regex
and bag-of-words baselines above are **now run** (both reach AUC 1.0;
`scripts/paraphrase_baselines.py`), so the deterministic paraphrase is
surface-separable and does not establish a semantic readout.
Adversarial, LLM-generated, larger multi-template, and
train-on-one-format/test-on-another paraphrase evaluations remain future work;
App. G.17 reports a template-based disjoint-vocabulary held-out check.

**On CodeGemma and DeepSeek**, the collapse at the reported
action-position cell is consistent with two readings: (i) those models'
$u_{\rm tx}$ directions track a *more specifically pytest-format*
signal than Qwen's, or (ii) the §4.3 reported action-position cell is
not the best cross-format readout cell on those models and the
format-readable direction lives elsewhere. App. G.13–G.14 below run a post-hoc layer ×
position sweep that supports reading (ii), but with the
selection-bias caveat that the cells were chosen on the
paraphrase AUC itself.

**Argmax behavior shifts under paraphrase (Qwen).** Under
realistic paraphrase, Qwen's first-token argmax distribution on
*fixed*-condition prompts shifts away from the original 83%
`grep` / 17% `edit` toward **41% `grep` / 17% `test` /
41% `edit`**, the prose summary apparently triggers a different
action prior than the pytest transcript even though the residual
projection still cleanly discriminates. CodeGemma argmax becomes
~100% `edit` on both conditions (collapse), DeepSeek argmax
becomes ~97% `view` on both (collapse). We surface this but do
not attempt to disentangle it; the projection-AUC result is
what's mechanistically interesting.

Outputs (per model and variant):
`results/`\allowbreak`monitor_real/`\allowbreak`{code,codegemma,deepseek}_tests_paraphrased_{minimal,realistic}_scores.json`;
caches: `cache-real-`\allowbreak`{qwen,`\allowbreak`codegemma,`\allowbreak`deepseek}-`\allowbreak`paraphrase-2026-05-19/`;
variants: `code_tests_paraphrased_{minimal,realistic}` in
`src/no_op_circuit/dataset/schema.py`.

## Cross-format layer sweep (post-hoc, selection-biased): post-hoc sweep identifies a CodeGemma L19 cell; no toy-derived pos=−1 DeepSeek cell

**Selection-bias caveat (applies to G.13 and G.14).** This and
the following appendix sweep many (layer, direction, format)
cells and report the highest-AUC cross-format discriminative cells. The
reported numbers are **selection-biased on the evaluation
formats** and lack held-out validation. They support the weak
claim that a cross-format-discriminative cell *can be found
under post-hoc cell selection* in each model's residual stream,
not the stronger claim that the
canonical methodology would have found it without seeing the
evaluation. We report them as exploratory.

G.12 above shows the canonical $u_{\rm tx}$ (toy-derived at the
§4.3 reported cell) is cross-format discriminative on Qwen but not on
CodeGemma or DeepSeek. This appendix asks the obvious follow-up:
does some *other* (layer, position) cell on the non-Qwen models
host a direction discriminative across the evaluated formats that
toy-substrate patching just didn't pick out? Three candidate directions per
layer L at
pos −1 (`scripts/cross_format_layer_sweep.py`):

- $v_{\rm toy}(L)$, paper-methodology direction: `mean(toy_fixed[L]) − mean(toy_buggy[L])`, fit on the 49 toy paired prompts at layer L. Frozen transfer to all three SWE-derived formats.
- $v_{\rm real}(L)$, `mean(real_pytest_fixed[L]) − mean(real_pytest_buggy[L])`, fit on the 499 SWE-derived pytest-format prompts at layer L (in-sample for pytest evaluation; cross-format for paraphrase). Here `real` is a legacy artifact name for the SWE-derived pytest-format prompt set, not an official SWE-bench agent evaluation.
- $v_{\rm para}(L)$, `mean(real_para_fixed[L]) − mean(real_para_buggy[L])`, fit on the 499 realistic-paraphrase prompts at layer L (in-sample for paraphrase evaluation; cross-format for pytest).

A "cross-format-selected" cell is one with high AUC on both
pytest and the deterministic realistic paraphrase (we use the
threshold pytest AUC > 0.85 AND realistic-paraphrase AUC > 0.85
to filter candidates).

**Result, CodeGemma-7B (28 layers, pos −1):**

| layer | $v_{\rm toy}$ on pytest | $v_{\rm toy}$ on para-realistic | $v_{\rm para}$ on pytest | $v_{\rm para}$ on para-realistic |
|------:|------------------------:|-------------------------------:|-------------------------:|---------------------------------:|
| 16 | 0.995 | 0.565 | 0.882 | 0.966 |
| 17 | 0.987 | 0.615 | 0.831 | 1.000 |
| 18 | 0.997 | 0.820 | 0.743 | 1.000 |
| **19**| **0.997** | **0.888** | 0.798 | 1.000 |
| 20 | 0.996 | 0.814 | 0.669 | 0.996 |
| 21 | 0.991 | 0.753 | 0.616 | 0.993 |
| 22 | 0.989 | 0.648 | 0.562 | 0.989 |
| 26 (§4.3 reported cell) | 0.950 | 0.456 | 0.511 | 0.981 |

**At L19 on CodeGemma, the canonical toy-derived direction
$v_{\rm toy}$ gives pytest AUC 0.997 AND paraphrase-realistic
AUC 0.888**, both higher than at L26 (the §5.1 headline cell:
pytest 0.950, paraphrase-real 0.456 / collapse). The §4.3 toy-substrate reported cell (L26) is selected by a different
criterion (toy F→B/B→F submargin shift), not paraphrase AUC;
L19 happens to give a better paraphrase result *and* a slightly
better pytest AUC on SWE-derived prompts, but the L19 cell was
selected post-hoc on the paraphrase format, so this is not
held-out evidence that L19 is a generally better mechanistic
anchor.

**Result, DeepSeek-Coder-1.3B (24 layers, pos −1):**

| layer | $v_{\rm toy}$ on pytest | $v_{\rm toy}$ on para-realistic | $v_{\rm para}$ on pytest | $v_{\rm para}$ on para-realistic |
|------:|------------------------:|-------------------------------:|-------------------------:|---------------------------------:|
| 11 | 0.963 | 0.572 | 0.990 | 0.931 |
| 14 | 0.962 | 0.627 | **0.994** | **0.951** |
| 17 | 0.925 | 0.566 | 0.994 | 0.958 |
| 18 | 0.912 | 0.552 | 0.992 | 0.968 |
| 22 (§4.3 reported cell) | 0.888 | 0.545 | 0.979 | 0.962 |

**No toy-derived $v_{\rm toy}$ direction is cross-format discriminative
on DeepSeek at any of its 24 layers**, best paraphrase-realistic
AUC under $v_{\rm toy}$ is 0.627 at L14, far below the 0.85
threshold. However, $v_{\rm para}$ (in-sample on realistic
paraphrase) at L11–L23 gives pytest AUC > 0.97, a direction
derived from paraphrased prompts transfers back to pytest at
near-perfect AUC. Thus, under post-hoc selection on these
evaluated formats, a paraphrase-fitted direction can be found in
DeepSeek that transfers back to pytest-format prompts; it just
isn't findable via toy pytest contrasts. (We avoid calling this an "abstract
pass/fail representation", the post-hoc cell selection and the
deterministic paraphrase mean both directions could be reading
format-specific lexical cues; G.12's caveats apply.)

![Cross-format layer sweep at pos −1 on CodeGemma-7B (top) and DeepSeek-Coder-1.3B (bottom). Blue: $v_{\rm toy}$ AUC on pytest format (the paper's canonical methodology). Red: same $v_{\rm toy}$ on realistic paraphrase. Green dashed: $v_{\rm para}$ (paraphrase-derived) transferred BACK to pytest. Dotted vertical line marks the §4.3 reported cell; on CodeGemma, the reported cell (L26) is NOT the highest cross-format-AUC cell (which sits at L19). On DeepSeek, no $v_{\rm toy}$ cell achieves both. Source: `scripts/cross_format_layer_sweep.py`.](figures/cross_format_layer_sweep.png){width=99%}

**Three things this resolves about the G.12 result:**

1. **A cross-format-discriminative cell can be identified post
 hoc on CodeGemma at L19.** The L26 reported cell is highly
 discriminative on pytest but collapses on the deterministic
 paraphrase; the L19 cell is selected post-hoc for cross-format
 AUC. Moving the CodeGemma `v_noop_cg` derivation to L19
 (keeping the toy contrastive methodology, just changing the
 layer) gives a cell with high AUC on both formats (pytest
 0.997, paraphrase-real 0.888). The L19 cell was selected on
 the paraphrase AUC itself, so this is exploratory evidence
 that such a cell *can be found under post-hoc cell selection*
 on CodeGemma, not held-out validation that the reported cell
 would have found it.
2. **No analogous v_toy cell exists on DeepSeek at pos −1.**
 No layer hosts a $v_{\rm toy}$ direction with paraphrase-real
 AUC > 0.85. A $v_{\rm para}$ direction (in-sample on
 paraphrase) does transfer back to pytest, but this is no
 longer the toy-substrate methodology and the in-sample fit
 prevents claiming an abstract representation.
3. **Methodological observation.** Contrastive mean-difference
 directions are sensitive to the format of the contrast data.
 Toy pytest contrasts find a direction discriminative on
 SWE-derived pytest-format prompts (by design); whether they find a
 format-general direction depends on the cell. Validating this
 methodologically would require multi-format toy contrasts
 evaluated on held-out paraphrases.

We do not retroactively change the §5.1 headline numbers (which
use the §4.3 reported patch cells as anchors and are honest to
the paper's primary methodology). The cross-format layer sweep
is reported here as **exploratory**: it suggests a cross-format-discriminative cell can be identified post hoc on CodeGemma, and
that an additional methodology (paraphrase or SWE-derived prompt
contrast) is needed to find one on DeepSeek, without claiming
either is a semantic pass/fail direction or that it generalizes
beyond the formats it was selected on.

Outputs:
`results/monitor_real/cross_format_layer_sweep_codegemma.json`,
`cross_format_layer_sweep_deepseek.json`.

## Cross-format position sweep (post-hoc, selection-biased): post-hoc-selected CodeGemma L19/pos=−4 and DeepSeek L16/pos=−5 cells

G.13 swept layers at pos=−1 only. The natural follow-up is the
**2D sweep**, does some non-canonical position host an even
better cross-format discriminative direction? For each
(layer L, position
p) in $\{-1, -2..., -8\}$, evaluate the three G.13 candidate
directions on all three formats
(`scripts/cross_format_position_sweep.py`).

**Result, CodeGemma-7B (28 × 8 = 224 cells).** The L19 finding
from G.13 (pos=−1) extends naturally to pos=−4, where the
discrimination strengthens:

| direction | best cell | pytest | para-real | para-min |
|-----------|--------------|-------:|----------:|---------:|
| $v_{\rm toy}$ | **L19 / pos=−4** | **0.976** | **0.939** | 0.680 |
| $v_{\rm toy}$ | L19 / pos=−1 (G.13)| 0.997 | 0.888 | 0.744 |
| $v_{\rm real}$ | L17 / pos=−4 | 1.000 (in-sample) | 0.967 | 0.844 |
| $v_{\rm para}$ | L19 / pos=−5 | 0.984 | 1.000 (in-sample) | 0.982 |

The cross-format-selected $v_{\rm toy}$ cell on CodeGemma is
(L19, pos=−4) with paraphrase-realistic AUC **0.939**, up
from 0.888 at pos=−1. The §4.3 reported patch cell (L26, pos=−1)
differs from this cross-format-selected cell on both axes (7
layers later, 3 positions earlier), but the reported cell was
selected by a different criterion (toy F→B/B→F submargin shift)
and we do not read the difference as the patching result being
"wrong", it measures a different target than cross-format
discrimination.

**Result, DeepSeek-Coder-1.3B (24 × 8 = 192 cells).** $v_{\rm toy}$
still does *not* recover on DeepSeek at any cell tested (best
para-real AUC under $v_{\rm toy}$ is **0.663** at L8/pos=−3).
But $v_{\rm real}$ (derived from the 499 SWE-derived pytest-format pairs
instead of the 49 toy pairs) **does** recover at pos=−5:

| direction | best cell | pytest | para-real | para-min |
|-----------|--------------|-------:|----------:|---------:|
| $v_{\rm toy}$ | best: L8/pos=−3 | 0.998 | 0.663 | 0.483 |
| **$v_{\rm real}$** | **L16 / pos=−5** | 1.000 (in-sample) | **0.925** | 0.736 |
| $v_{\rm real}$ | L10 / pos=−5 | 1.000 (in-sample) | 0.920 | 0.770 |
| $v_{\rm para}$ | L11 / pos=−5 | 1.000 | 1.000 (in-sample) | 0.886 |

**Post-hoc diagnosis (cells selected on paraphrase AUC; no
held-out validation):**

- **CodeGemma**: a $v_{\rm toy}$ cell at (L19, pos=−4) is
 cross-format discriminative at AUC 0.939. The §4.3 reported cell
 (L26, pos=−1) was selected by a *different* criterion
 (toy F→B/B→F submargin shift), not paraphrase AUC, so the
 cell mismatch is unsurprising.
- **DeepSeek**: no $v_{\rm toy}$ cell tested at pos in {−1, …,
 −8} achieves paraphrase-real AUC > 0.67. A $v_{\rm real}$
 direction (derived from the 499 SWE-derived pytest-format pairs, not 49
 toys) at (L16, pos=−5) achieves 0.925, but this is no longer
 the same methodology as the canonical toy contrast, the
 "fix" is to switch *both* the cell and the contrast data.

![Cross-format position-sweep heatmaps. Cell color = paraphrase-realistic AUC under each direction; black × marks the §4.3 canonical (L_site, pos=−1) reported cell; black circle marks the highest paraphrase-realistic AUC cell. Top row: CodeGemma; bottom row: DeepSeek. Left column: $v_{\rm toy}$; right column: $v_{\rm real}$. On CodeGemma, $v_{\rm toy}$ achieves high AUC at the post-hoc selected L19/pos=−4 cell (0.939). On DeepSeek, no $v_{\rm toy}$ cell tested reaches the 0.85 threshold; $v_{\rm real}$ achieves high AUC at the post-hoc selected L16/pos=−5 cell (0.925). Source: `scripts/cross_format_position_sweep.py`.](figures/cross_format_position_sweep.png){width=99%}

**Mechanistic reading (speculative).** The post-hoc-selected
cells with high evaluated-format AUC on CodeGemma and DeepSeek
sit at a
non-canonical position (−4 / −5) at mid-stack layers (L17–L21).
The action-token position (−1) at the toy-contrast reported cell picks up
a more *specialised* downstream readout site that is more
format-specific. One reading is that earlier positions (still
inside the prompt tail, before the action-token output) host a
less format-bound representation of the test evidence. We do not
have held-out validation that this reading is correct rather
than an artifact of post-hoc cell selection.

**Sample-size matters too.** The 49-pair toy contrast finds a
cross-format discriminative cell on Qwen (L24, pos=−1) and
CodeGemma (L19, pos=−4) but not on DeepSeek. A 499-pair
SWE-derived pytest-format contrast finds one on DeepSeek (L16, pos=−5).
Suggests the toy-substrate methodology may be under-powered for
some models; scaling to a SWE-derived prompt contrast addresses it on
DeepSeek. Both findings are post-hoc and need held-out
validation.

**Three-model status after G.13 + G.14:**

\begin{center}
\begin{tabular}{l p{0.46\linewidth} l r}
\toprule
model & status under post-hoc cross-format search & selected cell / dir.\ & AUC \\
\midrule
Qwen & Recovered at canonical Qwen causal cell & L24/$-1$, $v_{\rm toy}$ & 0.995 \\
CodeGemma & Recovered only after post-hoc layer/position search (toy contrast) & L19/$-4$, $v_{\rm toy}$ & 0.939 \\
DeepSeek & Not recovered with toy contrast; only after a SWE-derived pytest-format contrast & L16/$-5$, $v_{\rm real}$ & 0.925 \\
\bottomrule
\end{tabular}
\end{center}

**The exploratory three-model picture** is that a cross-format-discriminative cell can be identified post hoc in each model's
residual stream, for Qwen at the canonical causal cell, for CodeGemma
only after a post-hoc layer/position search (the canonical cell is not
the best cross-format readout cell), for DeepSeek only after switching
contrast data. We do **not** claim the
post-hoc cells generalize to held-out paraphrase formats or to
other transcript distributions; they were selected on the same
formats used to evaluate them. Established more rigorously, this
would require: (i) cross-validated cell selection on one
paraphrase format, evaluation on another; (ii) keyword-regex /
BoW baselines at the same cells to distinguish format-abstract
readout from format-keyword readout; (iii) larger held-out
paraphrase suites.

Outputs:
`results/monitor_real/cross_format_position_sweep_codegemma.json`,
`cross_format_position_sweep_deepseek.json`.

## Temporally-separated transcript (Qwen): the direction carries the transcript verdict forward; it beats a stateless turn-local regex, not a full-history or stateful parser

Every result so far places the transcript *adjacent* to the
decision: the pytest output is the last evidence block before
the `Action:` token. A natural objection is that the mechanism
is then trivial: the model is reading a `FAILED` token we
inserted deterministically, one block above the readout. This
appendix tests whether the L24/pos −1 direction still carries
the pass/fail transcript verdict when the transcript is **positionally
distant** from the decision, separated by intervening agent
turns whose content is identical across buggy and fixed.

**Design.** We reformat each of the 499 SWE-derived prompts into a
multi-turn agent trace on Qwen (`code_tests_stale_multiturn` in
`src/no_op_circuit/dataset/schema.py`):

| turn | role | content |
|------|------|---------|
| 1 | user | issue + code + "what action?" |
| 2 | assistant | `Action: test` |
| 3 | user | **the condition-matched pytest transcript** + "what action?" |
| 4 | assistant | `Action: grep` |
| 5 | user | `grep -rn "TODO"` → `(no matches found)` + "what action?" |
| 6 | assistant | `Action: view` |
| 7 | user | `CHANGELOG.md` (generic) + "what action?" |
| (decision) | assistant | `Action: ` $\leftarrow$ residual + action logits read here |

Turns 4–7 are **byte-identical across conditions**. The prompt
still contains buggy/fixed code in turn 1, but the format-matched
no-transcript control below shows that this code difference is
not discriminative for the frozen L24/pos −1 linear readout
(AUC 0.509); the additional signal in the 0.807 result is
therefore attributable to the upstream transcript at turn 3. We
score with the **same frozen toy-trained unit monitor direction $u_{\rm tx}$**
at (L24, pos −1) used in §5.1, no retraining.
The **format control** (`code_multiturn_notranscript`) uses the
identical scaffold with the turn-3 transcript replaced by a
neutral placeholder (`(no test output available)`), holding
everything constant (including the turn-1 code) except the
discriminative transcript.

Two regex baselines per prompt: a **turn-local** regex
(`"FAILED"` in the *decision-point local context*, the last
user turn, turn 7) and a **full-scrollback** regex (`"FAILED"`
anywhere in the rendered prompt). Server-side scoring returns
only per-prompt projections + action logits + regex flags
(`modal_app/multiturn_experiment.py`); CIs are paired-bootstrap
(B=10000, seed=0).

**Result.**

| setting | transcript position | projection ROC-AUC | turn-local regex AUC | full-scrollback regex AUC |
|--------------------------------|--------------------|---------------:|---------------:|---------------:|
| §5.1 (adjacent) | last block before decision | **0.989** | 1.000 | 1.000 |
| **`code_tests_stale_multiturn`** | **turn 3, behind 2 turns** | **0.807** [0.788, 0.827] | **0.500** | 1.000 |
| `code_multiturn_notranscript` (control) | absent (format-matched) | 0.509 [0.494, 0.525] | 0.500 | 0.500 |
| §G.9 (1-turn, no transcript) | absent | ≤ 0.52 | – | – |

**The transcript verdict survives the temporal separation.** With the
transcript pushed two agent turns + the intervening tokens
upstream, and the decision-point's local context containing
**no pass/fail tokens at all**, the frozen direction still
discriminates the failing-transcript (buggy) from the
passing-transcript (fixed) condition at **AUC 0.807** (CI excludes
both chance and the format control). The signal degrades from
the adjacent 0.989 (dilution across the intervening turns is
expected) but stays far above the **0.509** format control and
the §G.9 no-transcript floor. The control's chance AUC confirms
the multi-turn *format* does not by itself create
discriminability, the 0.807 is the carried-forward transcript
signal.

**This beats a stateless turn-local regex (only).** A
**stateless turn-local** regex, one inspecting only the current
observation before deciding the next action, reads
**chance (0.500)**, because the decision-local context contains
no `FAILED` token: it is two turns upstream. The residual
projection carries the transcript verdict to the decision point where a
stateless local parser sees nothing. It does **not** beat a
**full-history** regex or a **stateful** parser that records
earlier test outcomes: a full-scrollback regex (re-scanning the
entire context) still recovers AUC 1.000 here, and a stateful
parser would simply store the turn-3 test-output verdict. The result's value
is therefore **mechanistic**: it shows the model carries the
upstream transcript verdict forward to the final action position, **not** deployment superiority over text parsing.

**The action hides what the residual reveals.** In the stale-
multiturn setting Qwen's first-token argmax distribution is
**identical across conditions**, buggy: 173 `edit` / 326 `grep`;
fixed: 173 `edit` / 326 `grep`, yet the projection separates the
two at 0.807. The emitted action carries *no* information about
the pass/fail transcript verdict here (consistent with the paper's 0% noop /
action-prior-dominance story); the residual carries it anyway.
This is the cleanest demonstration in the paper that the
direction reads evidence the behavior does not surface.

**Scope.** This is **Qwen-only** and a **single forward pass**
over a constructed multi-turn prompt, not a live agent loop with
KV-cache continuation or context eviction; "temporally
separated" here means positionally distant within one rendered
context, with the transcript still textually present (hence the
full-history regex still recovers it). A genuinely *evicted*
transcript (truncated out of the context window) cannot be read
from a single-pass residual, that is the §G.9 no-transcript
regime, which is negative. The contribution is **mechanistic**:
the L24 direction carries the upstream transcript verdict to the final
action position, which rebuts the adjacent-keyword-triviality
reading. We do **not** claim operational superiority over text
parsing: the residual beats only a stateless turn-local regex,
not a full-history regex or a stateful parser that records test
outcomes as they appear.

Outputs: `results/monitor_real/multiturn_experiment_*.json`;
variants `code_tests_stale_multiturn` /
`code_multiturn_notranscript` in
`src/no_op_circuit/dataset/schema.py`;
`modal_app/multiturn_experiment.py`, `scripts/analyze_multiturn.py`.

## Cross-model contradictory-transcript control: full 2 × 2 on CodeGemma and DeepSeek

§5.2's contradictory-transcript control, the keystone "reads
transcript, not code" result, was originally Qwen-only. This
appendix runs the same control on CodeGemma-7B and DeepSeek-1.3B with the
**identical** 2 × 2 (code × transcript) design and statistics.

**Method.** For each of the 499 SWE-bench-Verified-derived tasks we forward all four
cells fresh on the GPU (server-side scoring, so only a small
per-cell JSON returns, no large-tensor download competing with
other transfers):

| cell | code | transcript | variant / condition |
|------|-------|------------|----------------------------------|
| BB | buggy | failing | `code_tests` / buggy |
| FF | fixed | passing | `code_tests` / fixed |
| BF | buggy | passing | `code_tests_swapped` / buggy |
| FB | fixed | failing | `code_tests_swapped` / fixed |

Each cell is scored with the model's frozen toy-trained unit monitor direction $u_{\rm tx}$
at its §4.3 reported patch cell (CodeGemma L26, the **all-49**
direction matching the §5.1 headline, ‖v‖=6.181; DeepSeek L22,
‖v‖=12.255), `score = −projection`. CodeGemma applies the same
2400-token cap as §5.1 (a few over-long cells drop, leaving
N=496 tasks present in all four cells; DeepSeek N=499). ΔCode,
ΔTranscript, the interaction, and bootstrap CIs (B=10000,
seed=0) use the identical formulas as
`scripts/contradictory_transcript_analysis.py`.

**Full 2 × 2 cell means** (score = −projection; higher = more
buggy-like):

| cell | CodeGemma-7B | DeepSeek-1.3B |
|------|-------------:|--------------:|
| (B,B) buggy code + failing tx | +8.307 | −13.164 |
| (B,F) buggy code + passing tx | +5.398 | −18.854 |
| (F,B) fixed code + failing tx | +8.338 | −13.104 |
| (F,F) fixed code + passing tx | +5.460 | −18.808 |

The two failing-transcript cells (B,B and F,B) cluster together
and the two passing-transcript cells (B,F and F,F) cluster
together, *regardless of the code*, exactly the signature of a
transcript-driven readout. The main-effect decomposition and
swapped-only AUCs are in §5.2's cross-model table; the headline
is that on both models ΔTranscript dominates (CodeGemma +2.894,
DeepSeek +5.697), |ΔCode| ≤ 0.06, and the swapped-only
transcript AUC (0.951 / 0.888) far exceeds the code AUC
(0.049 / 0.112). Argmax-action distributions per cell are in the
output JSON.

Outputs:
`results/monitor_real/contradictory_crossmodel_20260520T034649Z.json`;
`modal_app/contradictory_crossmodel.py`,
`scripts/analyze_contradictory_crossmodel.py`; CodeGemma all-49
direction `results/v_noop_codegemma_all49.pt`. The Qwen §5.2
analysis uses `results/monitor_real/contradictory_transcript_2x2.json`,
`scripts/contradictory_transcript_analysis.py`, the
`code_tests_swapped` variant + flip_tests scaffolding in
`src/no_op_circuit/dataset/schema.py` and
`src/no_op_circuit/agent/prompt.py`, and Modal cache run
`results/cache-real-qwen-swap-n500-20260518T074930Z/`.

## Held-out paraphrase robustness: Qwen positive, CodeGemma/DeepSeek negative at reported cells

The §G.12 deterministic paraphrase is keyword-leaky: a regex tuned to its
vocabulary reaches AUC 1.0. We run a stricter **template-based held-out**
robustness test on Qwen. We author two *train* paraphrase templates and two
*held-out* templates (A, B) and check overlap automatically: under
lowercase / `[a-z]+`-tokenization / stopword-and-structural-word stripping,
there is **no shared discriminative content vocabulary** (empty intersection;
train words `{wrong, right, diverged, matched, breaks, fulfills...}` vs
held-out `{violation, contradicted, honored, open, closed, discrepancy,
consistent...}`); functional words and template structure may still leak weak
cues. We render the 499 SWE-derived paired prompts under each family (499 buggy
+ 499 fixed per family), score the **frozen** toy-trained Qwen $u_{\rm tx}$ at
L24/pos −1 with **no** held-out cell/threshold/direction selection, and compare
against text baselines fit **only** on train-template text. Positive class is
the matched failing-transcript/buggy condition.

\begin{center}
\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{l c r r r r}
\toprule
classifier & fit on held-out? & held-out A AUC & held-out B AUC & pooled AUC & pooled AP \\
\midrule
frozen Qwen $u_{\rm tx}$ (L24/pos $-1$) & no & 0.905 & 0.983 & \textbf{0.943} & 0.932 \\
literal-\texttt{FAILED} regex & no & 0.500 & 0.500 & 0.500 & 0.500 \\
train-vocabulary keyword rule & no & 0.500 & 0.500 & 0.500 & 0.500 \\
BoW logistic (train$\to$held-out) & no & 1.000$^\ast$ & 1.000$^\ast$ & 0.750 & 0.833 \\
\bottomrule
\end{tabular}}
\end{center}

$^\ast$The per-template BoW AUC is **degenerate**: a text baseline sees only
each template's two constant transcript strings (four distinct strings total),
so per-template AUC merely reflects whether those two strings are ordered
correctly. The residual monitor's per-template AUCs are not degenerate in the
same way because the residual varies across full prompts, whereas the text-only
BoW baseline sees only two transcript strings per held-out template. The
**pooled** column is therefore the fair head-to-head. (Frozen
$u_{\rm tx}$ reaches AUC 0.994 on the train-template family; its mean
failing-minus-passing classifier-score gap on held-out is +2.60 on A and +3.82
on B.) The frozen direction generalizes to disjoint-content-vocabulary held-out
paraphrases at pooled AUC 0.943, above the literal-token and train-vocabulary
keyword baselines (both exactly at chance, as expected when the discriminative
words are held out) and above the train-fit BoW (pooled 0.750). The BoW result
should be interpreted cautiously rather than dismissed: it orders each held-out
template's two strings (per-template AUC 1.0, the degenerate within-template
comparison above) but its cross-template calibration is inconsistent (pooled
AUC 0.750, held-out prediction spread 0.108), consistent with weak
functional-word / template-structure residue rather than robust lexical
transfer. It is therefore the strongest no-held-out-fitting text baseline we
evaluated in this template setup, but it should not be read as a
general-purpose held-out paraphrase baseline. This is **preliminary,
template-based** evidence of held-out paraphrase robustness *on Qwen only*;
we do **not** claim a semantic pass/fail abstraction, because the held-out set
is two hand-authored deterministic templates (not adversarial or
LLM-generated).

**Cross-model reported-cell follow-up (CodeGemma, DeepSeek): the held-out robustness is
Qwen-specific.** We rerun the identical templates and text baselines on
CodeGemma and DeepSeek, scoring each model's **frozen toy-trained direction at
its §4.3 reported cell** (CodeGemma L26/pos −1, all-49 direction; DeepSeek
L22/pos −1) with **no post-hoc layer/position search**. The text baselines are
model-independent (they see only the transcript text): literal-`FAILED` 0.500,
train-vocabulary keyword 0.500, BoW (train→held-out) pooled 0.750 throughout.
For each model, the pooled held-out set contains 499 paired tasks split across
held-out templates A and B (250 and 249 tasks respectively), i.e. 998 total
prompts; the pooled held-out AUC pools templates A and B, and the per-template
"held-out A / B" columns are computed over the 250 and 249 matched
failing/passing pairs, respectively. All three models use the full 499 paired tasks
here with no token-cap drops (the short paraphrase prompts do not hit the cap
that reduced CodeGemma to 497 in §5.1). Audit:
`scripts/audit_heldout_paraphrase_counts.py` →
`results/heldout_paraphrase_robustness/count_audit.{json,md}`.

| model (reported cell) | residual train-template AUC | residual held-out pooled AUC | held-out A / B |
|---|---:|---:|---:|
| Qwen (L24/pos −1) | 0.994 | **0.943** | 0.905 / 0.983 |
| CodeGemma (L26/pos −1) | 0.917 | 0.649 | 0.645 / 0.660 |
| DeepSeek (L22/pos −1) | 0.567 | 0.619 | 0.792 / 0.454 |

Only Qwen's frozen direction generalizes to the held-out templates (0.943,
above the BoW 0.750). CodeGemma's direction holds on its own train templates
(0.917) but degrades sharply on held-out (0.649, below the model-independent
BoW 0.750); DeepSeek's collapses (held-out pooled 0.619, and *below chance* on
template B), and is weak even on train templates (0.567). This matches the §G.12
finding that the CodeGemma/DeepSeek directions are more pytest-format-specific,
and confirms that held-out paraphrase robustness here is a **Qwen-specific**
property of the reported-cell direction, not a cross-model one. CodeGemma's
0.917 train-template AUC is measured on the new G.17 train-template family; it
should not be compared directly to the G.12 deterministic realistic-paraphrase
cell where the reported-cell direction inverted.

**Held-out operating points (threshold frozen on train templates).** Fixing
each model's balanced-accuracy threshold on its train-template prompts and
applying it unchanged to the held-out templates (CPU reanalysis of the same
score artifacts, no model forwards;
`scripts/held_out_threshold_calibration_crossmodel.py` →
`results/monitor_real/held_out_threshold_calibration_crossmodel.{json,md}`)
gives a true out-of-sample operating point. Qwen holds (held-out precision
0.868, recall 0.900, fixed-condition FPR 0.136, accuracy 0.882), whereas
CodeGemma (precision 0.609, recall 0.571, FPR 0.367, accuracy 0.602) and
DeepSeek (precision 0.611, recall 0.611, FPR 0.389, accuracy 0.611) fall to
near-chance operating points, consistent with their collapsed held-out AUC. This
is the paraphrase-setting held-out calibration and is not a substitute for a
§5.1 canonical-menu calibration table: the raw §5.1 per-task projection caches
are not retained locally, so reproducing the canonical-menu random-split /
leave-one-repo-out classifier table for CodeGemma and DeepSeek would require
re-running model forwards, which we leave as future work.

Artifacts and
the exact preprocessing / vocabulary-overlap report are in App. J
(`results/heldout_paraphrase_robustness/{qwen,codegemma,deepseek}_summary.json`).

# SAE decomposition: full analysis

![**SAE decomposition (exploratory) on Qwen.** *(A)* Cumulative top-k OMP ablation of the original seed-unknown artifact reduces the buggy/fixed margin gap by 0% at k=1, +26.4% at k=2, and reaches +34.0% at k=3 (95% paired-bootstrap CI shaded); the magnitude is seed-fragile (App. H.7 reports +30.3% at k=2 under seed=0, plateau ending at k=5). *(B)* Argmax-action distribution before vs after OMP top-8 ablation on 296 paired buggy/fixed prompts: clean Qwen prefers `grep` (~84%); ablating the OMP top-8 features flips **80.1%** of prompts (237/296), 236/237 `grep → edit`, one `grep → test`, an action-prior override toward `edit`, not `noop`. The §5.2 control re-interprets the ablated features as pass/fail-transcript-text features. Both numbers are seed-fragile (App. H.7); CodeGemma replication: geometric pattern holds, behavioral specificity does not (App. H.6).](figures/main_sae.png){#fig:sae width=98%}

## Setup and geometric finding

**Corpus accounting.** The SAE corpus uses three different counts
of "prompt" that we summarize here once to avoid confusion:

| unit | count | role |
|------------------------------------------|---------:|-----------------------------------------------------|
| Paired toy tasks | **49** | substrate for §4 patching, steering, $u_{\rm tx}$ derivation |
| Toy `(task, variant)` cells in SAE cache | varied | training input is residual positions, not prompts |
| Total prompts in SAE training corpus | **1282** | union of all cached prompts across all variants |
| Residual positions in training corpus | **1.23M**| streaming SAE training operates on these |
| Paired `(task, variant)` pairs with both `buggy` and `fixed` materialized | **148** | used for the H.2–H.4 ablation evaluation |
| Paired buggy/fixed prompts in ablation eval | **296** | = 148 × {buggy, fixed} |

The arithmetic 49 × 5 variants × 2 conditions = 490 prompts is the
theoretical maximum if every `(task, variant)` cell had both
conditions materialized. The actual cache had 1282 prompts because
some variants were materialized multiple times during pipeline
iteration (older prompt-template runs in the cache root). Of
those, 148 `(task, variant)` cells have both `buggy` and `fixed`
versions, and those 296 prompts (148 × 2) are the ablation
evaluation set.

**Caveat on the SAE training corpus.** Because the SAE analysis
is treated as **exploratory** (App. H.7 establishes the precise
plateau magnitudes are seed-fragile), we did **not** deduplicate
the 1282-prompt cache before training, and we did not re-train
on a clean 296- or 490-prompt corpus. The cache may contain
multiple materializations of the same `(task, variant, condition)`
under slightly different prompt-template versions used during
pipeline iteration. None of this paper's causal claims (§4) or
monitor claims (§5.1–§5.2, App. G) depend on this corpus; they
use only the residual at L24/pos −1 from a clean re-run.
A clean-corpus SAE re-train remains future work.

We trained a TopK SAE [@gao2024scaling; @bricken2023monosemanticity;
@templeton2024scaling] on Qwen2.5-Coder-1.5B L24 `resid_pre`
activations cached over the 1282-prompt toy corpus, 1.23M
residual positions, streaming. Architecture: d_in = 1536,
d_sae = 4096, k = 16. The SAE reached explained variance
EV = 0.976 with 16.0 mean active features and a 0.02% dead-feature
rate after 8 epochs on a Modal A10G (73 s training, ~$0.02). This
corpus was chosen after a generic Python-only SAE
(d_sae = 24,576, k = 32, EV = 0.954 on `bigcode/the-stack-smol`)
reconstructed the contrast direction with cosine only +0.21: the action-position
residual after a chat-templated agent prompt is out of distribution
for mid-Python-file activations.

**The contrast direction is geometrically dense in the SAE basis.** On the task-distribution SAE the encoder TopK reconstruction of the contrast direction is poor
(cos = +0.10 at k = 16, the trained sparsity). To separate "the encoder
is suboptimal for OOD directions" from "the direction is intrinsically
dense in this basis," we evaluate **orthogonal matching pursuit
(OMP)**, a greedy k-sparse reconstruction procedure that
iteratively selects decoder columns and refits coefficients by
least squares (greedy, not globally optimal). The
"OMP-then-ablate" sequence is the manual analog
of automated-circuit-discovery methods such as ACDC
[@conmy2023acdc]. OMP needs k ≈ 128 features to reach cos ≥ 0.80 (task
SAE 0.806; generic SAE 0.838) and k ≈ 256 to clear cos ≥ 0.85 (task
0.900; generic 0.947) on either SAE. The basis itself spans the full
input space (dense least-squares cosine = 1.000), so this is not a
capacity bug, $u_{\rm tx}$ simply does not admit a small-k sparse
expansion. Encoder-TopK underperforms OMP by 0.2–0.4 cosine at every
k, indicating that the trained encoder biases are tuned to typical
residuals (norm ~30) and degrade for steering directions of small
magnitude ($\|v_{\rm raw}\|$ = 5.89).

## Cumulative top-k ablation on the original SAE artifact

A fine-grained cumulative sweep over k ∈ {1, 2, …, 8, 16, 32, 128}
re-runs the same ablation pipeline at every cut. The reduction-vs-k
curve is sharply non-linear:

| k | reduction % | 95% CI (B=10K) | Wilcoxon p |
|----:|------------:|----------------------|-----------:|
| 1 | −0.10% | [−0.29%, +0.00%] | 0.84 |
| 2 | **+26.4%** | [+20.4%, +32.7%] | 5.3e-13 |
| 3 | **+34.0%** | [+26.4%, +42.0%] | 1.1e-13 |
| 4 | +34.0% | [+26.4%, +41.8%] | 1.1e-13 |
| 5 | +33.9% | [+26.3%, +41.7%] | 1.3e-13 |
| 6 | +33.9% | [+26.3%, +41.7%] | 1.3e-13 |
| 7 | +33.9% | [+26.3%, +41.8%] | 1.3e-13 |
| 8 | +33.9% | [+26.2%, +41.6%] | 1.3e-13 |
| 16 | +37.7% | [+23.7%, +52.0%] | 1.8e-06 |
| 32 | +37.5% | [+23.7%, +51.3%] | 1.9e-06 |
| 128 | +37.4% | [+23.8%, +51.3%] | 2.1e-06 |

The reduction is essentially zero at k = 1, jumps to +26% at k = 2,
clears the +34% ceiling at k = 3, and is **flat at +34% from k = 3
through k = 8**. The smallest OMP subset reaching the +34% plateau is
**three features, not eight**, i.e. the ablation curve saturates by
k = 3. We avoid calling this a "sufficient set" because a +34%
margin-gap reduction is partial, not full behavioral sufficiency;
"smallest tested subset reaching the observed plateau" is the
defensible phrasing. The argmax-level override result (H.4) is
measured at the k = 8 plateau endpoint; we have not separately
measured the k = 3 argmax-flip rate.

## Specificity controls: two random-8 baselines

| set | reduction % | across-seed 95% CI | Mann-Whitney p vs OMP top-8 |
|-------------------------------|-------------|--------------------|------------------------------|
| OMP top-8 | **+33.11%** | [+25.4%, +40.9%] | – |
| random-8 (any, 10 seeds) | +0.00% | [−0.08%, +0.08%] | **$3.0 \times 10^{-96}$** |
| random-8 (firing≥5, 10 seeds) | −0.85% | [−9.3%, +7.3%] | **$3.9 \times 10^{-17}$** |

The Mann-Whitney p-values are computed over per-prompt gap shifts
pooled across the 10 random draws (296 tasks × 10 seeds vs the
296 OMP per-task reductions), not over the 10 seed-level means;
the extreme values reflect the large pooled sample, and the
across-seed 95% CI column is the operative measure of
seed-to-seed variability. We therefore treat these p-values as
descriptive rather than as the primary robustness evidence; the
seed-level interval is the relevant summary of seed-to-seed
variability.

The first baseline draws from {0,…,4095}\\OMP_top128 uniformly; per-task
gap shifts are essentially zero on every seed because TopK sparsity (k = 16
per position) means a random feature draw rarely overlaps the firing set
on any given prompt. The **firing-only baseline** is the tighter
control: drawing 8 random features from the 45 actively-firing features
outside OMP top-128 gives a per-seed reduction distribution between
−21.7% and +19.6%, averaging to −0.85% across 10 seeds. Only OMP's
signed-coefficient-aware selection produces consistent positive
reduction.

## Action-level effect of OMP top-8 ablation

Argmax shifts per (task, condition) under OMP top-8 ablation on Qwen:

| condition | view | grep | test | edit | noop |
|----------------------------|-----:|-----:|-----:|-----:|-----:|
| **buggy**, clean | 0.0% | **83.8%** | 0.0% | 16.2% | 0.0% |
| **buggy**, OMP top-8 | 0.0% | 5.4% | 0.7% | **93.9%** | 0.0% |
| **fixed**, clean | 0.0% | **83.1%** | 0.0% | 16.9% | 0.0% |
| **fixed**, OMP top-8 | 0.0% | 1.4% | 0.0% | **98.6%** | 0.0% |

Ablating the OMP top-8 SAE features flips the argmax on **237 of 296
prompts (80.1%)**; **236/237 flips are `grep → edit`** and one is
`grep → test` (no other transitions). We read this as an
**action-prior override** under the present prompt format, the
default `grep` hedge is dislodged toward `edit`, not toward `noop`, rather than as a clean abstention readout. It is not threshold
calibration (the change is at the argmax, not at a downstream
scoring step), but neither is it evidence that the OMP top-8 set
is *the* abstention-driving feature set; in light of §5.2 the
features are better read as pass/fail-transcript features
(re-interpretation in H.5).

We measure the argmax flip at OMP top-8 because the +34% margin-reduction
plateau is reached at k = 3 and held through k = 8 (H.2); reporting the
action-level effect at the plateau endpoint avoids overstating
sufficiency. The k = 3 argmax-flip rate is not measured in this work.

## Top-5 OMP features have plausible pass/fail-transcript interpretations

- **F2669** (OMP coef −1.10, contrib +0.236): promotes ' error',
 ' Error', ' errors'. $u_{\rm tx}$ *subtracts* this, attenuating error-attention
 is consistent with the pass/fail-transcript interpretation.
- **F3129** (coef −0.58, contrib +0.118): promotes ' traceback',
 ' Trace'. $u_{\rm tx}$ again *subtracts* this.
- **F3171** (coef +0.75, contrib +0.099): promotes ' already',
 ' Already' and suppresses ' corrected', ' corrections'. $u_{\rm tx}$ *adds*
 this, boosting "already done" semantics.
- **F2950** (coef +1.17, contrib +0.130): logit-lens suppresses
 `\tedit`, `\tview`, `\tgit`, leading-tab action tokens. $u_{\rm tx}$ *adds*
 this, boosting action-token suppression.
- **F1954** (coef +1.81, contrib +0.206): fires almost exclusively on
 '#' (Python comment marker).

![SAE feature decomposition of $u_{\rm tx}$ (Qwen, L24 resid_pre, task-distribution TopK SAE with d_sae=4096, k=16, EV=0.976 over 1.23M positions).](figures/sae_decomposition.png){width=98%}

![Cumulative OMP top-k ablation curve on Qwen (N=148 (task, variant) pairs from the 49-task toy substrate; 296 prompts with both `buggy` and `fixed` materialized).](figures/sae_topk_curve.png){width=85%}

## Cross-model SAE replication on CodeGemma (partial, uses the 20-task responsive-subset direction)

**Direction used.** This exploratory CodeGemma SAE replication
uses the 20-task responsive-subset `v_noop_cg` (the
direction used in the §4.2 steering plot, not the all-49
direction used for the §5.1 monitor headline). The clean
CodeGemma toy gap reported below (+2.589 logits) is the
responsive-subset gap, not the all-49 gap of +1.347 reported in
§4.3. We retain the analysis as exploratory SAE evidence; the
all-49 CodeGemma SAE was not rerun. The 20-task subset is
therefore used in: (i) the §4.2 steering plot, and (ii) this
appendix; the §5.1 monitor headline and §4.3 cross-model table
both use the all-49 direction.

We repeated the entire pipeline on CodeGemma-7B-it at L26/pos −1
using the (responsive-subset) frozen v_noop_cg direction. The
**geometric** finding replicates; the **behavioral** specificity
finding does **not**.

**Setup.** We cached 1.20M positions of L26 resid_pre over the 1282
task prompts and trained a TopK SAE with d_sae = 8192, k = 48 (linear
scaling from Qwen's d_sae/d_in = 2.67 ratio; we initially trained at
k = 24 matching Qwen's ratio but reached only EV = 0.75, below our
gate, so retrained at k = 48). The k = 48 SAE reached **EV = 0.830**
with 0.0% dead features (in the [0.80, 0.85] caveat band).

**Geometric result.** OMP k = 128 → cos 0.677, k = 256 → cos 0.779,
k = 1024 → cos 0.969, same "resist-sparsity" character as Qwen.

**Behavioral result.** Running the per-feature-set ablation pipeline
on the 146 `(task, variant)` pairs that fit under the 2400-token cap
(2 of 148 dropped for length):

| set | mean gap | reduction | 95% CI | Wilcoxon p |
|---------------------|----------|-----------|-------------------|------------|
| clean (CodeGemma) | +2.589 | – | – | – |
| sae_recon (control) | +2.589 | 0.0% | – | – |
| ablate OMP top-8 | +2.425 | **+6.4%** | [−3.2%, +16.4%] | **0.103** |
| ablate OMP top-32 | +3.178 | −22.7% | [−34.4%, −11.1%] | 1.00 |
| ablate OMP top-128 | +2.972 | −14.8% | [−25.4%, −4.2%] | 1.00 |

On CodeGemma the OMP top-8 reduction is **+6.4%** (CI includes zero,
p = 0.10, not significant), and the larger subsets *increase* the
buggy-fixed gap by 15–23%. The behavioral specificity claim that
holds on Qwen does not hold on CodeGemma. Three candidate
explanations, none isolable without further compute: (i) CodeGemma's
SAE was weaker (EV 0.830 vs Qwen 0.976); (ii) v_noop_cg was derived
from a smaller paired-task substrate (N = 20 vs 49 on Qwen); (iii)
CodeGemma's L26 may not encode the edit-vs-noop signal as a clean
low-rank composition the way Qwen's L24 does, even though both
sites are at corresponding relative depth and the monitor transfers
cleanly to both.

**One CodeGemma feature is interpretable.** Feature 6974 (OMP rank
2, coef +1.05, contribution +0.170) promotes ` passed`, ` OK`,
` pass`, a pass/fail-transcript feature (the test-passing analog
of Qwen's "passing-test transcript" feature), with positive
v_noop_cg contribution. The other top-5 features have lower
logit-lens interpretability (top promotions/suppressions are rare
non-English tokens), likely reflecting CodeGemma's larger
vocabulary and the more polysemantic features expected at 7B
scale.

![CodeGemma-7B SAE feature decomposition of v_noop_cg (L26 resid_pre, task-distribution TopK SAE with d_sae=8192, k=48, EV=0.830 over 1.20M positions).](figures/sae_decomposition_codegemma.png){width=98%}

## Seed=0 re-seed: geometric finding replicates; behavioral sparse effect is qualitatively consistent but not exact

The headline SAE artifact for §5.4 / H.1–H.5 pre-dates seed pinning
(see App. J.4); the specific feature indices (F1954, F2669, F2950,
F3129, F3171) are not bit-reproducible. We re-train the same
architecture under `--seed 0` on the same 1.23M-position task
corpus and re-run OMP-k=128. The seed=0 cumulative-ablation
re-run uses a **broader 1096-prompt toy+SWE-derived evaluation set**
(not the 296-prompt toy-only ablation set of H.2–H.4), so the
comparison below is **qualitative** rather than an exact
replication of the H.2 296-prompt ablation curve. The new
artifact: `results/sae/qwen_l24_resid_pre_TASK_d4096_k16_seed0.pt`
(EV = 0.962, l0 = 16.0, 0.0% dead, 97 s training). Top-5 OMP
features by signed contribution: F1304, F3166, F639, F945, F3902, entirely different indices, as expected for different SAE seeds.

**Geometric finding replicates.** $u_{\rm tx}$ reconstruction cosine at
OMP-k=128: **0.812** (vs **0.806** in §H.1), within sampling
noise. The "dense in basis, OMP needs k ≈ 128 for cos ≥ 0.80"
qualitative claim is seed-robust.

**Behavioral-sparse finding: qualitatively consistent re-run.**
Cumulative OMP-top-k margin-gap reduction
(`scripts/ablation_stats.py` on
`results/sae/ablate-distributed-seed0/`):

| k | original (§H.2) | seed=0 | comment |
|----:|----------------:|-------------:|----------------------------------|
| 1 | −0.10% | **+25.7%** | different feature ordering |
| 2 | +26.4% | **+30.3%** | similar magnitude |
| 3 | +34.0% | +30.3% | plateau onset matches |
| 4 | +34.0% | +30.2% | plateau holds |
| 5 | +33.9% | +30.2% | plateau holds |
| 6 | +33.9% | **−41.6%** | inversion: gap WIDENS |
| 7 | +33.9% | −41.6% | |
| 8 | +33.9% | −40.3% | through k=8 plateau does NOT hold |
| 16 | +37.7% | +3.9% | |
| 32 | +37.5% | +3.9% | |
| 128 | +37.4% | −72.6% | |

What survives: at small k, OMP selection of features aligned with
$u_{\rm tx}$ reduces the buggy-fixed margin gap by ~30% (seed=0; was
~34% in the original artifact), with a sharp non-linear onset
between k=1 and k=2-5. The original-artifact statement that the
smallest tested subset reaching the plateau is three features
should therefore be read as artifact-specific rather than
seed-robust: under seed=0 the plateau is reached at **k=2** at +30.3%
(CI [+26.8%, +33.8%], Wilcoxon p ≈ $10^{-45}$), and the plateau width
is much narrower (k=2-5 vs k=3-8). The +34% number itself is a
seed-specific artifact; the qualitative "small-k subset achieves
substantial gap reduction" claim is robust, but the **precise
plateau magnitude and width depend on the SAE seed**.

What does *not* survive: the "plateau holds through k=8" claim.
Under seed=0, ablating the OMP-top-6 through OMP-top-8 features
*widens* the buggy-fixed gap (negative reduction). This is
consistent with the OMP procedure: features ranked 6-8 in seed=0
do not have the same role as in the original artifact (different
encoder learns a different basis), and ablating them can shift
the residual in the opposite direction. The §H.4 "OMP-top-8 flips
80% of argmaxes, nearly all `grep → edit`" finding therefore cannot be
read as a stable property of the SAE basis; it is a property of
the specific (seed-unknown) artifact used in §H.4 / §5.4 / Fig. \ref{fig:sae}.

**How to read §5.4 / §H.2-H.5 in light of this.** The qualitative
findings (the contrast direction is dense in basis; a small OMP-selected feature
subset achieves a non-trivial, ~30%, gap reduction; the
reduction has a sharp non-linear onset between k=1 and k=2-3)
replicate. The quantitative headline numbers (+34.0% margin
reduction at OMP-k=3, 80.1% argmax flip at OMP-k=8, the specific
F-indices in §H.5) **do not replicate exactly** under a re-seeded
SAE and should be read as artifact-specific. The semantic logit-lens interpretations in §H.5 (error / traceback / already)
similarly depend on the artifact; in light of §5.2 the right
interpretation is anyway pass/fail-transcript features, not "no-op
circuit elements."

Outputs:
`results/sae/v_noop_features_DISTRIBUTED_seed0.json`,
`results/`\allowbreak`sae/`\allowbreak`ablate-`\allowbreak`distributed-`\allowbreak`seed0/`\allowbreak`ablation_results.json`,
`results/`\allowbreak`sae/`\allowbreak`ablate-`\allowbreak`distributed-`\allowbreak`seed0/`\allowbreak`ablation_stats.json`.

# In-context attribution within the cached last-32-token window

On the existing toy cache the residual stream is stored only for the
**last 32 token positions** of each prompt. Across all 49 paired
toys the last-32-token sequence is identical: 13 tokens of canonical
question text, 12 tokens of chat-template close, and 7 tokens of
`Action: ` suffix. Any buggy-vs-fixed differential in the L24
resid_pre projection onto $u_{\rm tx}$ at a given offset is therefore
**purely attentional information flow** from the earlier (varying)
test-output and code content.

**Per-offset projection** (per-offset paired Wilcoxon p < $10^{-10}$ at
every position):

| section | offset range | mean(buggy) | mean(fixed) | differential |
|-------------------------------|--------------|------------:|------------:|-------------:|
| question text | [−31, −19] | +2.25 | +3.24 | **+0.99** |
| chat-template close | [−18, −7] | +2.36 | +3.10 | **+0.74** |
| `Action: ` suffix | [−6, 0] | −0.94 | +2.40 | **+3.34** |

The differential grows monotonically as we approach the Action
position, from ≈+1 logit-unit at offset −31 to **+5.9 at offset 0**.
The per-token Action: suffix differential (+3.34) is **3.4× stronger**
than the question-text section (+0.99). The $u_{\rm tx}$ signal is present
throughout the prompt tail but **crystallizes sharply over the final
seven positions**.

![In-context attribution of $u_{\rm tx}$ on Qwen toy substrate (N=49 paired tasks).](figures/incontext_attribution.png){width=98%}

# Reproduction

## Bootstrap

The source repository and small analysis artifacts are public; the large
activation caches are archived at the Hugging Face dataset listed below.

```bash
git clone https://github.com/faizancodes/no-op-circuit-paper.git
cd no-op-circuit-paper && python -m venv .venv && source .venv/bin/activate
pip install -e ".[analysis]"
cp .env.example .env # fill in MODAL_TOKEN_ID, MODAL_TOKEN_SECRET,
 # HF_TOKEN, OPENROUTER_API_KEY
```

## Cost table

Tracked Modal compute before the final consistency-pass experiments was
approximately **$17** (the
original ~$8.91; plus ~$4 of paraphrase-cache jobs across three
models for App. G.12; plus ~$4 of later server-side scoring jobs
for the temporal-separation control (App. G.15, Qwen) and the
cross-model contradictory-transcript control (App. G.16,
CodeGemma + DeepSeek). The cross-format layer/position sweeps and
the all-49 CodeGemma recompute are pure local computation on
previously cached residuals, $0 added Modal cost.) OpenRouter
(LLM-generated tasks + flaky transcripts): **~$6**; approximate total
cloud cost **~$23** (approximate cloud-cost figures, not an audited bill;
the later consistency-pass jobs below were not separately tracked, so the
true total is slightly higher). The SWE-derived peak-cell patching job
(Qwen, N=200, three cells; §4.1), the toy discrete five-action patching
job (Qwen, N=49; §5.6), and the cross-model five-action follow-ups
(CodeGemma-7B on A100, N=49; DeepSeek-Coder-1.3B single-token, N=49;
§5.6) were added after the earlier cost accounting and were not
separately tracked. The CodeGemma vs §4.3 consistency audit
(`scripts/audit_patching_consistency.py`) is pure local CPU
computation, $0 added Modal cost. Phase-by-phase breakdown in the project repository
(long-form per-experiment cost table; this appendix collapses it
to the headline). The action-menu controls (action-order / binary /
abstract-label / letter-only on Qwen; action-order / binary on
CodeGemma (A100) and DeepSeek; the DeepSeek single-token rerun) and the
noisy-transcript monitor run are small first-token-scoring jobs added
later; their cost was not separately tracked but is on the order of a
few dollars total. The five-action steering decomposition, the BoW /
paraphrase baselines, and the tokenization audit are pure local
computation ($0 added Modal cost).

## Entry points

All Modal entry points expose `--model`, `--layer`, and `--run-id`
flags. The three core jobs:

```bash
# Cache residual streams for all 49 tasks × variants × {buggy, fixed}
modal run -m modal_app.cache_dataset --model <hf-slug> \
 --variants issue_only,code,code_tests

# Bidirectional activation patching (peak-cell heatmaps)
modal run -m modal_app.patch_dataset --model <hf-slug> \
 --variant code_tests --max-suffix 2 --layer-step 2 --bidirectional

# Single-direction steering (computes the raw contrast vector, sweeps α)
modal run -m modal_app.steer_dataset --model <hf-slug> \
 --cache-dir results/cache-<RUN_ID> \
 --variant code_tests --layer <peak> --position -1
```

Local analysis scripts (`scripts/run_monitor_real.py`,
`scripts/ablation_stats.py`, `scripts/codegemma_per_repo_calibration.py`,
`scripts/incontext_attribution.py`, `paper/figures/render_all.py`)
regenerate all monitor metrics and figures from the cached `.pt`
files and the trained v_noop / SAE artifacts under `results/`.

The consistency-pass experiments add these entry points:
`scripts/check_action_tokenization.py` (tokenization audit);
`modal_app/action_order_control.py` + `scripts/run_action_order_control.py`
+ `scripts/analyze_action_order_control.py` (action-menu / position-balance
controls, including the DeepSeek single-token rerun via
`--experiment action_order_custom --action-words view,find,test,edit,done`);
`scripts/analyze_five_action_decomp.py` (five-action steering decomposition,
no GPU); `modal_app/noisy_monitor.py` + `scripts/analyze_noisy_monitor.py`
(noisy-transcript monitor vs regex vs bag-of-words);
`scripts/paraphrase_baselines.py` (keyword / BoW paraphrase baselines, no GPU);
and `modal_app/swe_peak_patching.py` +
`scripts/analyze_swe_peak_patching.py` (Qwen SWE-derived peak-cell causal
patching at L24/pos −1 with L12/pos −1 and L24/pos −8 controls, all five
action logits; doubles as a discrete five-action patching decomposition on
the 49 toys via `--tasks toy --output-dir results/five_action_decomp`,
including A100 dispatch via `--gpu a100` for CodeGemma-7B and a custom
single-token vocabulary via `--action-words view,find,test,edit,done
--abstain-word done` for the DeepSeek single-token follow-up at L22/pos −1).
Outputs:
`results/swe_peak_patching/qwen_swe_peak_patch_{scores,summary}.json`;
`results/five_action_decomp/qwen_patching_five_action_{scores,summary}.json`
(Qwen);
`results/five_action_decomp/deepseek_swe_peak_patch_{scores,summary}_toy_single_token.json`
(DeepSeek single-token follow-up); and
`results/five_action_decomp/codegemma_swe_peak_patch_{scores,summary}_toy.json`
(CodeGemma, pending consistency reconciliation with the §4.3 May-2026
artifact `results/patch-codegemma_7b_it-20260516T031403Z/`; see
`scripts/audit_patching_consistency.py` and
`results/consistency_audit/patching_consistency_audit.json` for the audit
output).

The **held-out paraphrase robustness** test (App. G.17) uses
`scripts/heldout_paraphrase_robustness.py`: `--self-test` checks
template-vocabulary disjointness on CPU; `--run` renders the 499 SWE-derived
paired prompts under each template family and scores the model's frozen
toy-trained direction at its reported cell via
`modal_app.noisy_monitor.score_monitor` (`score_monitor_a100` for the 7B
CodeGemma); `--reanalyze` recomputes all per-template AUC/AP, the
train→held-out BoW prediction spread, the score gaps, and the
lowercase/`[a-z]+`-tokenization/stopword-stripped vocabulary-overlap report
from the cached scores on CPU (no new forwards). It is run cross-model at each
model's §4.3 reported cell with **no post-hoc layer/position search**, via the
`--model`, `--site-layer`, `--direction-path`, `--output-prefix`, and `--gpu`
flags. The cells and frozen toy-trained directions are: Qwen L24/pos −1
(`results/`\allowbreak`steer-20260516T021522Z/v_noop.pt`); CodeGemma L26/pos −1
(all-49 direction `results/`\allowbreak`v_noop_codegemma_all49.pt`, scored on
A100); DeepSeek L22/pos −1
(`results/`\allowbreak`steer-deepseek-coder-13b-instruct-`\allowbreak`20260517T012848Z/v_noop.pt`;
legacy artifact path name, model is DeepSeek-Coder-1.3B).
Outputs
`results/`\allowbreak`heldout_paraphrase_robustness/`\allowbreak`{qwen,codegemma,deepseek}_{summary,scores}.json`;
each summary JSON records the model, site, direction artifact, exact
preprocessing, and vocabulary-overlap report. The text baselines
(literal-`FAILED`, train-vocabulary keyword, train→held-out BoW) are
model-independent and are fit only on train-template text.

**§5.6 five-action decomposition, renderer reconciliation (detail moved
from §5.6).** The §5.6 table is a five-action *sanity check* under the
current decomposition renderer (`modal_app/swe_peak_patching.py`), not a
re-estimate of the §4.1/§4.3 causal-localization numbers. The §4.1/§4.3
estimates (Qwen toy clean B−F gap +0.659, F→B +0.648, 43-task bidirectional
+0.69/+0.64) come from the original paper renderer; that toy `code_tests`
*patch* artifact is not available locally (only the `code` negative-control
variant is) and is not in the HF archive `faizancodes/no-op-circuit-caches`
(which holds the SWE-derived residual caches, not the toy patch grid), so a
per-task prompt-hash / margin reconciliation could not be completed. The
current-renderer Qwen F→B shift (+0.625) is close to §4.1's +0.648, but the
current-renderer clean B−F gap (+0.532) is lower than §4.1's +0.659, the same
fixed-condition renderer-drift direction found for CodeGemma. We therefore keep
the §4.1/§4.3 numbers as the localization estimates and present §5.6 only as a
sanity check (reconciliation:
`scripts/`\allowbreak`reconcile_qwen_five_action.py` and
`scripts/`\allowbreak`audit_qwen_five_action_consistency.py`; outputs
`results/`\allowbreak`consistency_audit/`\allowbreak`qwen_five_action_reconciliation.{json,md}`
and `qwen_five_action_consistency_audit.json`). For CodeGemma, the audit
(`scripts/`\allowbreak`audit_patching_consistency.py`) finds matching action
token IDs and task IDs but a systematic newer-renderer fixed-condition drift
(new − old buggy diff mean +0.08, median 0.0; fixed diff mean −1.31, median
−2.0; new B−F gap mean +2.73 vs old +1.35); this is a renderer-comparability
issue for the newer five-action decomposition, not a correction to the §4.3
reported-cell result, so CodeGemma is omitted from the §5.6 table while its
qualitative `noop`-never-induced result (0/49 clean and patched) stands.

## Reproducibility: pinned random seeds

All scripts and Modal entry points accept `--seed` (default `0`) and
seed `torch`, `torch.cuda`, `numpy`, and `random` from a single source.
`torch.use_deterministic_algorithms(True, warn_only=True)` is enabled
so non-deterministic CUDA kernels emit warnings instead of silently
diverging. The pinned scripts:

| script | seeded via |
|-----------------------------------------------------|---------------------------|
| `modal_app/train_sae.py` | `--seed` |
| `modal_app/ablate_sae_features.py` | `--seed` |
| `modal_app/ablate_random_features.py` | `--base-seed` (per-draw) |
| `scripts/train_probes.py` | `--seed` |
| `scripts/run_monitor_real.py` | `--seed` |
| `scripts/ablation_stats.py` | `--seed` (pre-existing) |
| `scripts/compute_auc_cis.py` (B4) | `--seed` |
| `scripts/adversarial_v_noop_baselines.py` (B6) | `--seed` |
| `modal_app/swe_peak_patching.py` | `--seed` (subset selection) |

**Caveat (the headline SAE artifact, now characterized by H.7).**
The Qwen TopK SAE referenced in §5.4 / Appendix H.5, `sae/qwen_l24_resid_pre_TASK_d4096_k16.pt`, EV 0.976, was trained
**before** seed pinning was added. App. H.7 now reports a seed=0
re-train of the same architecture on the same corpus and compares
the results directly. The **geometric finding** replicates
($u_{\rm tx}$ reconstruction cosine at OMP-k=128 is 0.812 vs 0.806). The
**behavioral-sparse finding** *partially* replicates: a small
OMP-selected feature subset still achieves ~30% margin-gap
reduction with a sharp non-linear onset between k=1 and k=2, but
the **precise +34.0% plateau magnitude and through-k=8 plateau
width do NOT replicate** (seed=0 plateau is +30.3% at k=2-5, with
the plateau ending at k=5 instead of holding through k=8). The
80%-argmax-flip number from H.4 is therefore artifact-specific
rather than seed-robust. The current `train_sae.py` is seed-pinned;
the seed-unknown artifact is bundled with the code release only to
preserve the literal F-indices in App. H.5 (which are themselves
artifact-specific; the seed=0 top-5 indices are entirely different).

## Activation caches (HuggingFace dataset)

The residual-stream activation caches that the monitor, control,
and sweep analyses run on (~66 GB of per-(task, variant,
condition) `.pt` tensors across the three models) are archived as
a public HuggingFace dataset:
**`huggingface.co/datasets/faizancodes/no-op-circuit-caches`**. The dataset
card documents the cache contents, its limitations as a non-source release, and
the prompt-derived metadata caveats (e.g. last-token text / token ids that may
contain short oracle-window fragments). The small
consistency-pass artifacts ship in the repo under `results/`:
`action_order_control/`, `five_action_decomp/`, `noisy_monitor/`,
`paraphrase_baselines/`, and `tokenization/`.
The small artifacts the analyses also need, the per-model frozen
`v_noop` directions (`results/steer-*/v_noop.pt`,
`results/v_noop_codegemma_all49.pt`), the patching manifests
(`results/patch-*/`), and the aggregate metric JSONs
(`results/monitor_real/`), are kept in the code repository
itself, so only the large tensor caches come from the Hub. The
aggregate metric JSONs are retained, but the canonical-menu per-task
projection records with the task/repository metadata needed to reproduce
the CodeGemma/DeepSeek held-out calibration splits are not retained
locally (§5.1, App. G.17); regenerating those canonical-menu calibration
tables would require rerunning model forwards or regenerating the missing
per-task score records.

Fetch the caches into `results/`:

```python
from huggingface_hub import snapshot_download
snapshot_download("faizancodes/no-op-circuit-caches",
 repo_type="dataset", local_dir="results")
```

or, for a single experiment's cache,

```bash
hf download faizancodes/no-op-circuit-caches \
 --repo-type dataset --local-dir results \
 --include "cache-real-qwen-n500-20260516T235301Z/**"
```

Cache → experiment map: `cache-20260515T221105Z` (Qwen toy
substrate, §4); `cache-{codegemma_7b_it,deepseek-toy}-*` (toy
substrates for the other two models, §4.3); `cache-real-{qwen,
codegemma,deepseek}-n500-*` (§5.1 monitor); `cache-real-qwen-
swap-n500-*` (§5.2 contradictory-transcript control);
`cache-real-{codegemma,deepseek}-paraphrase-*` (App. G.12
paraphrase); `sae/` (App. H SAE weights + ablation outputs). The
upload + verification scripts are
`scripts/upload_caches_to_hf.py`,
`scripts/upload_caches_retry.sh` (resumable wrapper), and
`scripts/verify_hf_upload.py` (per-file size + sha256 check).

The dataset is primarily residual-activation tensors and run
manifests, but the per-prompt records also include
prompt-derived metadata (the last-32 token ids and last-token
text), which can contain fragments of the oracle code window;
see §8 (Ethics and Data Use) for the SWE-bench data-use caveats that apply.
