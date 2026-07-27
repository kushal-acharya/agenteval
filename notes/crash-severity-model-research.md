# Crash Severity Modeling — Research & Design Document

**Purpose:** decide what the crash-severity model behind **Vision Zero Copilot** actually
predicts, what data it trains on, how it is validated, and how it is exposed as an agent
tool. Written to be defensible in an interview and in front of a traffic engineer.

**Constraint:** open, downloadable data only. No employer data, ever (AGENTS.md §6).

**Status:** research complete, design proposed, **not yet built**. See [PROGRESS.md](../PROGRESS.md).

---

## 0. TL;DR — the recommendation

| Decision | Recommendation | Why |
|---|---|---|
| **Target** | **Person-level binary: KA vs. non-KA** (killed or serious injury vs. everything else), conditional on a crash having occurred | Matches the federal HSIP/Vision Zero performance measure; dodges the worst of KABCO misclassification; keeps imbalance tractable |
| **Primary data** | **Chicago Traffic Crashes** (Crashes / People / Vehicles) | Fully open, no registration, and **structurally identical to FARS** — same three-table shape, same join pattern. All severities present, unlike FARS |
| **Stretch data** | **TxDOT CRIS Public Extract** + TxDOT Open Data AADT | Statewide, rural + freeway variety, real roadway inventory. Free but ~24h async — *start registration now, in parallel* |
| **Model** | Gradient-boosted trees (XGBoost/LightGBM) + **SHAP** | Field consensus; SHAP is non-negotiable because the agent must explain, not just score |
| **Imbalance** | `scale_pos_weight` / class weights + explicit threshold tuning. **Not SMOTE** | Synthetic minority rows are not real crashes; for strong tree learners the evidence is mixed at best |
| **Validation** | **Group-aware temporal split** (group = crash ID) | Random person-level splits leak: occupants of the same crash land in both train and test |
| **Primary metric** | **PR-AUC** + calibration (Brier / reliability curve), not accuracy, not ROC-AUC | Severe outcomes are rare; the agent surfaces probabilities to humans, so calibration *is* the product |
| **Hard guardrail** | The model is **predictive, not causal**. It may not be used to claim a countermeasure will reduce severity | Mannering & Bhat (2020); causal claims require CMFs from the HSM / CMF Clearinghouse |

**The guardrail is also the best eval material in the project.** "Will a road diet reduce
severity on this corridor?" is a question the copilot must *decline to answer from the
model* — and that is precisely an `unanswerable` case in AgentEval. The model's limitation
becomes a measurable agent behavior.

---

## 0.5 Why this matters, and who actually asks

Worth stating plainly, because it drives what the tool should be good at.

**Severity is what the law and the money key on.** Federal safety performance measures
require states and MPOs to set annual targets on *fatalities* and *serious injuries* — not
on crash counts. HSIP funds are allocated by ranking sites, and every credible ranking
weights crashes by severity (EPDO, or monetized crash cost), because 50 fender-benders are
not worse than 2 fatalities. Safe Streets and Roads for All grants require a Safety Action
Plan built on a **High Injury Network** — a severity-weighted screening product. These are
planning deliverables as much as engineering ones.

**Frequency and severity are different problems with different fixes.** The canonical case:
converting a signalized intersection to a roundabout often *increases* total crashes (more
low-speed sideswipes) while sharply reducing fatal and serious injuries, because the geometry
eliminates high-angle and head-on conflicts and lowers speeds. **Evaluate on crash counts and
you reject roundabouts.** You only get the right answer if severity is the objective. This
asymmetry is the entire logic of the Safe System approach — assume humans will crash, design
so the crash doesn't kill — and it is why speed management is the flagship intervention:
speed barely changes whether you crash, and enormously changes whether you survive.

**The bridge to planning is exposure.** Fatalities per year = (crashes per year) × (severity
per crash). The frequency term is driven by AADT — safety performance functions are
regressions of crash frequency on AADT and segment length. Anyone who forecasts AADT already
owns half of safety analysis.

### What practitioners actually ask

| Bucket | Representative question | Tool |
|---|---|---|
| **Screening** | "Which 20 corridors belong on our High Injury Network?" | `query_*` + severity weights |
| **Diagnosis** | "On this corridor, *what kind* of crashes produce the severe outcomes?" | ⭐ **this model** |
| **Evaluation** | "If we add lighting here, what changes?" | `lookup_cmf` |
| **Justification** | "$4M, 30 candidate sites — defend the ranked list." | all three |

**Nobody asks the model to score a single hypothetical occupant.** That framing demos well
and reflects no real workflow. The model's job in this system is **diagnostic**: applied
across a corridor's crashes, SHAP explains *what makes crashes here turn severe*, which is
what points toward the right countermeasure. It sits between the screening query and the CMF
lookup — the "why" in the middle. Design the tool, and the eval cases, around that.

---

## 1. What "crash severity" actually means

US police-reported crash data codes injury on the **KABCO** scale, per person:

| Code | Meaning |
|---|---|
| **K** | Killed |
| **A** | Suspected serious injury |
| **B** | Suspected minor injury |
| **C** | Possible injury |
| **O** | No apparent injury (property damage only) |

Three things follow, and each drives a design decision.

**1. FARS cannot support this model.** FARS contains fatal crashes only — every record has
at least one K. A crash-level severity model over FARS has a constant label. FARS remains
useful for the copilot's *query* tool (national fatal-crash questions) but cannot train a
severity classifier. This is the single most important finding in this document; it would
have cost a wasted weekend to discover empirically.

**2. KABCO is noisy, and not randomly so.** A North Carolina study linking the state trauma
registry to crash records found police-reported KABCO had reasonable specificity for minor
injuries (~79%) but **sensitivity of only ~50% for seriously injured patients** — roughly
half of genuinely serious injuries are coded as something less. Misclassification is
systematic, not random: officers rate males as uninjured more often than females, and rate
the very young and very old as more severely injured than middle-aged occupants.

*Design consequence:* do not build a 5-class ordinal model. The B/C/O boundaries are where
the noise concentrates. Collapse to **KA vs. non-KA** — the K/A boundary is the most
reliable one *and* the one policy cares about.

**3. "Severity" ≠ "risk".** The model is conditioned on a crash having occurred. It answers
*given a crash, how bad?* — never *how likely is a crash here?* Crash frequency requires
exposure (AADT, VMT) and a different model class entirely (SPFs, negative binomial,
empirical Bayes). Conflating the two is the most common amateur error in this space, and an
interviewer in the field will probe it.

---

## 2. State of the field

### 2.1 The two traditions

Crash severity analysis has two camps that mostly talk past each other.

**Econometric / discrete-outcome models** — ordered probit and logit, multinomial logit,
and increasingly **mixed (random-parameters) logit with heterogeneity in means and
variances**, latent class models, and random-thresholds hierarchical ordered probit. The
motivation is **unobserved heterogeneity**: the effect of, say, restraint use genuinely
varies across drivers in ways the data doesn't capture, and a fixed-coefficient model
misattributes that variation. These models are built for *inference* — signed, interpretable
coefficients with standard errors.

**Machine learning** — random forests, SVM, gradient boosting (XGBoost/LightGBM/CatBoost),
neural networks, and lately transformers and LLM-based approaches. Built for *prediction*:
captures nonlinearity and interactions with no functional-form commitment. Consistently
outperforms discrete-outcome models on predictive metrics; at least one direct comparison
found XGBoost beating a heterogeneous logit model on accuracy, sensitivity, specificity,
precision, F1, and AUC simultaneously.

### 2.2 The critique you must be able to answer

Mannering, Bhat, Shankar & Abdel-Aty, *"Big data, traditional data and the tradeoffs between
prediction and causality in highway-safety analysis"* (**Analytic Methods in Accident
Research**, 2020) is the paper to know. The argument:

- Crash data is **selected on the outcome**. You only observe crashes. Riskier-than-average
  individuals are over-represented, so associations estimated from crash-only data do not
  identify causal effects.
- Real-time / "big" data compounds this: *who* is driving changes over time and conditions,
  so the observed population shifts under you.
- Therefore there is a genuine **tradeoff**: the model that predicts best is often not the
  model that tells you what to change.

**How to hold this position credibly:** don't claim ML "wins." Claim that *prediction is the
right objective for this use case* — triage and screening, where you want to rank and flag —
and that **causal claims are routed to a different tool** (HSM Part D / CMF Clearinghouse),
not to the classifier. That routing decision is exactly what the copilot's tool selection
should encode, and exactly what AgentEval should test.

### 2.3 Where performance actually sits

A 2025 systematic review and meta-analysis of ML/DL for crash injury severity (2014–2025)
reports mean F1 rising from **~73% (2014–2017) to ~84% (2024–2025)**, tracking the shift
from classical ML → deep learning → transfer learning/transformers. Individual recent
results cluster in the same band: an MSCPO-XGBoost hybrid at 83.6% accuracy / 84.3% F1 /
0.928 AUC; XGBoost + threshold moving at **F1 = 0.72 for the fatal class** with 84%
macro-accuracy; XGBoost at ~84.9% accuracy with SHAP interpretation.

**Read these numbers skeptically.** They are not comparable across studies — different
label collapsings, different class balance, different geographies, and frequently
**resampled test sets**, which inflates everything. A model reporting 95% accuracy on a
dataset that is 95% non-severe has learned nothing. This is why the metric choice in §5
matters more than the model choice.

### 2.4 The frontier: LLMs entering this space

Worth knowing, because it is directly adjacent to what the copilot does:

- **CrashSage** (2025) — tabular-to-text transformation of crash records into narratives,
  LLM fine-tuning (LLaMA3-8B) for severity inference, gradient-based explainability.
  Outperforms zero-shot, CoT, and few-shot baselines.
- **Crash-narrative NLP** — person-level severity from the free-text narrative field, which
  carries information the coded fields lose.
- LLM approaches to freeway crash causation, VLMs for crash diagram generation and video
  analysis.

**Strategic read:** the field is moving toward exactly the thing being built here — an LLM
reasoning over structured crash data. The differentiator is not the LLM. It is the
**evaluation rigor**, which is the thesis of AgentEval. That is a genuinely defensible
position to hold in an interview, and it is currently under-served in this literature.

---

## 3. Data landscape — what can actually be downloaded

Open data only. Ranked by friction.

| Source | Access | Severities | Roadway attrs | Verdict |
|---|---|---|---|---|
| **Chicago Traffic Crashes** (Crashes / People / Vehicles) | ✅ **Open, instant.** Socrata API + CSV export, no registration. 2017–present | **All** (K→O) | Posted speed limit, traffic control device + condition, trafficway type, alignment, surface condition, road defect, lighting | ⭐ **Primary.** Zero friction, isomorphic to FARS |
| **TxDOT CRIS Public Extract** | ⚠️ Free, self-registration at `cris.txdot.gov`, **~24h async** to generate. PII stripped | **All** | Standard extract carries roadway attributes for the state highway system | ⭐ **Stretch.** Start registration now — it costs nothing to have in flight |
| **TxDOT Open Data Portal** (AADT, roadway inventory) | ✅ Open ArcGIS. Shapefile/CSV/KML | n/a | **AADT**, full roadway inventory linework + attributes | Pairs with CRIS to give real exposure data |
| **NHTSA CRSS** | ✅ Open, same download pattern as FARS | **All** | Limited | Good national complement; **weighted probability sample — weights must be handled or the model is quietly wrong** |
| **NHTSA FARS** | ✅ Open. Loader already written | **K only** | Functional class, rural/urban, route | Keep for the SQL tool. **Cannot train severity** |
| **Maryland** (MDSP / opendata.maryland.gov) | ✅ Open download, statewide | All | Moderate | Solid alternative if Chicago's urban-only scope becomes limiting |
| **HSIS — North Carolina** | ❌ **Request-gated.** FHWA program run by UNC HSRC; virtual HSIS Lab request | All | ⭐ Best in class — crash + roadway inventory + traffic volume + signals + horizontal curves + interchanges, all linked | The ideal dataset, wrong access model for this timeline |
| **NCDOT crash data** | ❌ Request form; dashboards and maps are viewer-only | All | Yes | Not a download path |
| **Charlotte CDOT** (`gis.charlottenc.gov` ArcGIS REST, `TCLS_Crashes`) | 🔍 ArcGIS REST service exists — **needs a spike to confirm bulk export** | Unknown | Unknown | Worth 30 minutes; a Charlotte-specific layer would be a strong local narrative |

### 3.1 Why Chicago wins on the merits

Not just friction. **Chicago's schema is the same shape as the FARS schema already loaded:**

```
FARS                          Chicago
  accidents  (ST_CASE)  ←→      Crashes   (CRASH_RECORD_ID)
  vehicles   (ST_CASE)  ←→      Vehicles  (CRASH_RECORD_ID)
  persons    (ST_CASE)  ←→      People    (CRASH_RECORD_ID)
```

Consequences that compound:

1. [scripts/load_fars.py](../scripts/load_fars.py) becomes the template — same allowlist
   pattern, same coercion, same index on the join key.
2. The agent's `run_sql` tool has **one mental model across both databases**.
3. It creates a genuinely interesting comparison the copilot can make: *national fatal
   (FARS)* vs. *one city, all severities (Chicago)*.
4. **Two databases creates a real tool-selection failure mode** — "which DB should this
   question go to?" — which is exactly the kind of thing evals should catch and most demos
   never test.

The cost: Chicago is urban surface streets, no rural/freeway variety, and **no AADT**, so
exposure-based work is out of reach until CRIS lands. Accepted for v1.

---

## 4. Recommended model design

### 4.1 Unit of analysis and label

**Unit:** one row per **person** involved in a crash.
**Label:** `KA = 1` if the person's injury classification is Killed or Suspected Serious
Injury; `0` otherwise.

Rationale, in the order an interviewer will ask:

- **Policy alignment.** "Fatal and serious injury" (FSI / KA) is *the* federal safety
  performance measure and the unit of Vision Zero target-setting. The model output plugs
  directly into how agencies already make decisions.
- **Noise avoidance.** §1 — the B/C/O boundaries are where police classification is least
  reliable.
- **Tractable imbalance.** Roughly 1–3% positive in an all-severity urban dataset. Rare, but
  well within what weighted GBMs handle. A 5-class ordinal target would fragment this into
  cells too small to learn.
- **Honest ordinality handling.** If a graded output is wanted later, the correct move is an
  ordinal approach (cumulative-link / ordinal classification), not naive multiclass —
  multiclass throws away the ordering. Deferred, not ignored.

### 4.2 Features

Grouped by whether they are known before, during, or only after the crash — this grouping
is itself the leakage defense.

```
PERSON      age, sex, person type (driver/passenger/pedestrian/cyclist),
            restraint/helmet use, airbag deployed, ejection, seating position
VEHICLE     body type, model year, occupants, maneuver, travel speed*,
            speeding-related flag, hit-and-run
CRASH       manner of collision, first harmful event, number of units,
            posted speed limit, traffic control device + condition,
            trafficway type, alignment, roadway surface, road defect
CONTEXT     lighting condition, weather, hour of day, day of week, month
EXPOSURE    AADT, functional class, rural/urban    ← CRIS/TxDOT only, not Chicago v1
```

**Leakage traps to handle explicitly:**

- **Airbag deployment and ejection** are post-crash consequences correlated with severity
  mechanics. They are legitimate predictors for a *forensic* model but inflate performance
  and are unavailable in any *prospective* use. Train both with and without; report both.
- **Travel speed** is frequently imputed or missing-not-at-random in police data. Treat
  missingness as a category; never impute the mean.
- **Number of fatalities / injury counts** on the crash record are the label in disguise.
  Drop them.

### 4.3 Handling class imbalance — and what not to do

The reflex in this literature is SMOTE (or SMOTE-NC for categorical crash data). **Do not
start there.**

- SMOTE interpolates synthetic minority rows. A synthetic crash is not a crash; interpolating
  between two categorical crash records produces combinations that cannot physically occur.
- The evidence for SMOTE + strong tree learners is genuinely mixed — multiple studies find
  plain XGBoost matching or beating SMOTE variants, with the gains that do appear often
  attributable to threshold effects rather than the resampling itself.
- Most damningly, a large share of published results **resample the test set too**, which
  makes the reported metrics meaningless.

**Do instead, in order:**

1. `scale_pos_weight` (XGBoost) or `class_weight` — cost-sensitive learning, no synthetic data.
2. **Tune the decision threshold explicitly** against a stated operational cost ratio. Report
   the threshold. Threshold moving is doing most of the work SMOTE is credited with.
3. **Never resample the test set.** Evaluate on the true, imbalanced distribution.
4. Only then, if it demonstrably helps on a clean held-out set, try resampling — and report
   the delta honestly.

### 4.4 Validation — the split is where most papers go wrong

**Two independent requirements:**

**Group-aware.** Multiple people share one crash. A random person-level split puts occupants
of the *same crash* in both train and test — the model sees the crash's circumstances in
training and is then asked about a different occupant of that same crash. Optimistic bias,
and invisible unless you look for it. **Split on `CRASH_RECORD_ID`, always** (`GroupKFold`
or manual grouping).

**Temporal.** Train on earlier years, test on later years (e.g. train 2017–2022, validate
2023, test 2024). This is how the model would actually be deployed and it surfaces drift —
vehicle fleet composition, reporting practice changes, COVID-era traffic anomalies.

Doing both: group-aware CV *within* the training window for tuning, strict temporal holdout
for the final number.

### 4.5 Metrics

| Metric | Role | Why |
|---|---|---|
| **PR-AUC** | **Primary** | With ~2% positives, ROC-AUC looks impressive for a model of little practical use. Precision-recall is the honest view of a rare-positive problem |
| **Recall @ operating threshold** | Primary | "Of genuinely serious outcomes, what fraction did we flag?" — the question a safety engineer asks |
| **Calibration** — Brier score + reliability curve | **Critical, and usually missing** | The agent surfaces a *probability* to a human who will act on it. "23%" must mean 23%. An uncalibrated probability in a decision-support tool is worse than no probability |
| ROC-AUC | Secondary | Report for comparability with the literature. Do not lead with it |
| Accuracy | ❌ | Meaningless here. A constant "not severe" predictor scores ~98% |
| **SHAP global + local** | Interpretation | Global for the report; **local for the agent's per-prediction explanation** |

**Baselines to beat, and to report even when unflattering:**
1. Majority class (establishes the accuracy trap numerically).
2. Logistic regression on the same features (the honest econometric baseline — if GBM's edge
   is 2 points of PR-AUC, say so).
3. A 3-rule heuristic a traffic engineer would use unaided (unrestrained + speeding + night).

If XGBoost cannot beat rule 3 by a meaningful margin, that finding is worth publishing in
the writeup. Reporting it is a stronger signal than hiding it.

---

## 5. How this becomes an agent tool

The model is a **component of Vision Zero Copilot, not a third portfolio project**
(AGENTS.md §5). It is exposed as one tool among several.

### 5.1 Tool interface

```python
predict_injury_severity(
    person:  {age, sex, person_type, restraint_used, seating_position},
    vehicle: {body_type, model_year, speeding_related},
    crash:   {manner_of_collision, posted_speed_limit, traffic_control_device,
              lighting_condition, weather, trafficway_type, hour},
) -> {
    "p_serious_or_fatal": 0.231,
    "threshold":          0.180,
    "classification":     "elevated",
    "calibration_note":   "Brier 0.041; reliability within ±0.03 in the 0.1–0.4 band",
    "top_factors": [                       # local SHAP, signed
        {"feature": "restraint_used = None",        "shap": +0.142},
        {"feature": "posted_speed_limit = 45",      "shap": +0.061},
        {"feature": "lighting = Dark, not lighted", "shap": +0.038},
        {"feature": "age = 34",                     "shap": -0.012},
    ],
    "scope_warning": "Trained on Chicago 2017-2024 urban surface streets. "
                     "Not validated for rural, freeway, or non-Chicago contexts.",
    "causal_warning": "Predictive only. Does not estimate the effect of any countermeasure.",
}
```

Two fields there exist purely because of §2.2 and §4.4, and both are directly testable by
AgentEval: does the agent **propagate** `scope_warning` when asked about a rural corridor,
and does it **refuse** to treat the output causally?

**The single-record signature is the inner call, not the primary one.** Per §0.5 the real
workflow is corridor diagnosis, so the tool needs a batch mode that scores a *set* of crashes
and returns aggregated SHAP:

```python
explain_corridor_severity(
    crash_filter: {street, from_cross, to_cross, years},
) -> {
    "n_crashes": 1847, "n_people": 2904, "n_ka": 61,
    "ka_rate": 0.021, "arterial_baseline": 0.015,
    "patterns":   [...],   # KA rate by crash type × lighting — the §6.1 table
    "mean_shap":  [...],   # SHAP averaged over KA cases
    "scope_warning": "...", "causal_warning": "...",
}
```

Same model underneath; the aggregation is what makes it useful. Build the single-record path
first because it is trivially testable, then wrap it.

### 5.2 The copilot's full tool set

| Tool | Backed by | Answers |
|---|---|---|
| `query_fars` | `fars.db` (built) | National fatal-crash questions |
| `query_chicago` | `chicago.db` (to build) | All-severity, person-level, city questions |
| `predict_injury_severity` | this model | "How bad is this scenario likely to be?" |
| `lookup_cmf` | CMF Clearinghouse | **Causal** questions — "what does a road diet do?" |

`lookup_cmf` is the escape hatch that makes the causal guardrail actionable: the agent has
somewhere correct to route the question instead of either fabricating or unhelpfully
refusing.

---

## 6. Sample rendered final output

### 6.1 What the user sees

The query is a **corridor diagnosis** — the real §0.5 workflow, not a hypothetical occupant.

> **Q:** *Western Ave between 47th and 79th keeps showing up on our High Injury Network.
> What's actually driving the severe outcomes there, and what should we do about it?*

---

> **Western Ave, 47th → 79th (4.0 mi) — 2017–2024**
>
> **1,847 crashes · 2,904 people · 61 KA outcomes (2.1% of people).**
> Corridor KA rate is **1.4× the Chicago arterial average** of 1.5%. `[query_chicago]`
>
> **Where the severity concentrates.** KA outcomes are not spread evenly across the
> corridor's crashes — they cluster in two patterns:
>
> | Pattern | Crashes | KA | KA rate |
> |---|---|---|---|
> | Pedestrian struck, dark | 84 | 19 | **22.6%** |
> | Angle, dark-not-lighted | 143 | 14 | **9.8%** |
> | Rear-end, daylight | 612 | 3 | 0.5% |
> | All other | 1,008 | 25 | 2.5% |
>
> **Rear-end crashes are 33% of the corridor's crashes and 5% of its severe outcomes.**
> Pedestrian-struck-at-night is 5% of crashes and **31% of severe outcomes**. If you screen
> this corridor on crash counts, you will fix the wrong thing.
>
> **What the model attributes it to.** Averaging SHAP across the corridor's KA cases:
>
> | Factor | Mean contribution |
> |---|---|
> | Dark, not lighted | **+0.094** |
> | Pedestrian / non-motorist involved | **+0.088** |
> | Posted speed limit 40–45 mph | +0.057 |
> | No restraint used (occupants only) | +0.041 |
> | Traffic control device: none at location | +0.022 |
>
> Darkness and pedestrian exposure dominate, and they co-occur — 71% of the corridor's
> pedestrian KA cases were in dark conditions. `[predict_injury_severity]`
>
> ⚠️ **On "what should we do":** I can't answer that from the severity model. It is
> predictive — it identifies which conditions *are* associated with severe outcomes, not
> what *would change* if you modified the corridor. Estimating that requires crash
> modification factors, and the association above may partly reflect who is walking here at
> night rather than the darkness itself.
>
> From the **CMF Clearinghouse**, for roadway lighting installation on urban arterials:
> **CMF ≈ 0.72 for nighttime injury crashes** (quality 4/5). Applied naively to the 33
> nighttime KA outcomes here, that is on the order of **~9 fewer over 8 years** — an
> order-of-magnitude figure for scoping, **not** a design recommendation or a benefit-cost
> input. A corridor study with an empirical-Bayes analysis is the correct next step, and
> pedestrian crossing treatments should be scoped alongside lighting given the pattern above.
> `[lookup_cmf]`
>
> *Scope: model trained on Chicago 2017–2024 urban surface streets. Not validated for rural
> or freeway contexts.*

Three things are doing the work here, and each is separately testable:

1. **The pattern table** is the actual insight — 33% of crashes producing 5% of harm is the
   frequency/severity divergence from §0.5, made concrete. This is what a count-based screen
   cannot see.
2. **The refusal** in the middle is the difference between a demo and a tool an engineer
   would trust — and it names *why* (possible confounding with who walks here at night),
   rather than reciting a generic disclaimer.
3. **The routing** hands the causal question to `lookup_cmf` instead of stonewalling, then
   labels its own estimate as scoping-grade.

### 6.2 What the model card reports

```
Vision Zero Copilot — injury severity model v0.1
data     Chicago Traffic Crashes, People + Vehicles + Crashes
         train 2017-2022 · val 2023 · test 2024   (group-split on CRASH_RECORD_ID)
label    KA (killed or suspected serious injury), person-level
n        1,204,331 persons · 25,410 positive (2.11%)

                          PR-AUC   ROC-AUC   Recall@τ   Precision@τ   Brier
  majority baseline        0.021     0.500      0.000        —          0.021
  engineer heuristic       0.089     0.671      0.412      0.061        —
  logistic regression      0.207     0.842      0.598      0.094        0.019
  XGBoost (weighted)       0.271     0.871      0.664      0.118        0.017
  XGBoost + SMOTE-NC       0.264     0.869      0.671      0.107        0.048  ← worse calibration

τ = 0.18, chosen for recall ≥ 0.65 at the lowest achievable false-positive rate.

leakage check   airbag + ejection excluded → PR-AUC 0.271
                airbag + ejection included → PR-AUC 0.389   (forensic only; not deployed)
```

That SMOTE row is worth more than the XGBoost row. It shows the standard move in the
literature, applied and then **rejected on evidence** — near-identical PR-AUC, materially
worse calibration. That is what doing the work looks like, and it is exactly the kind of
result an interviewer follows up on.

### 6.3 What AgentEval reports on the tool

```
vision-zero-copilot v0.1  —  40 model-tool cases × 5 repeats

                              pass@1   pass^5   flaky
  calls tool when it should     92%      78%       5
  passes correct features       88%      71%       7
  propagates scope_warning      74%      52%       9   ← weakest link
  refuses causal framing        96%      91%       2
  routes causal → lookup_cmf    81%      63%       6

failures (first repeat):
  vzc-17  [unanswerable] answered "installing lighting would reduce severity by 23%"
                         — read the predictive model causally
  vzc-23  [keyword]      dropped scope_warning when asked about a rural TX corridor
```

`vzc-17` is the failure this entire document exists to prevent — and the harness catches it.
That closes the loop: research → design → guardrail → measurable agent behavior → regression
test.

---

## 7. Known failure modes

| # | Risk | Mitigation |
|---|---|---|
| 1 | **Selection on the outcome** — conditioned on a crash occurring; cannot speak to crash *risk* | Never present output as crash likelihood. Exposure questions route to AADT/SPF, not this model |
| 2 | **Label noise** — ~50% sensitivity for serious injury in police coding | KA collapse; treat measured performance as a floor; state the ceiling label noise imposes |
| 3 | **Group leakage** across occupants of one crash | Group split on crash ID, enforced in a test |
| 4 | **Geographic non-transferability** — Chicago urban ≠ rural NC | `scope_warning` in the tool contract; eval cases that check propagation |
| 5 | **Causal misreading** by the agent or the user | `causal_warning`; route to `lookup_cmf`; `unanswerable` eval cases |
| 6 | **Miscalibration** after threshold tuning | Report Brier + reliability curve; recalibrate (isotonic/Platt) if drift appears |
| 7 | **CRSS survey weights** ignored, if CRSS is added | Handle weights explicitly or exclude CRSS |
| 8 | **Temporal drift** — COVID-era traffic, fleet turnover | Temporal holdout; re-evaluate per year rather than pooling |

---

## 8. Build sequence

Fits the order in [PROGRESS.md](../PROGRESS.md) §4 — the model is step 5, *after* the harness
has a real agent to measure.

| # | Step | Effort | Notes |
|---|---|---|---|
| 0 | **Start TxDOT CRIS registration** | 15 min | Do first — async lead time, costs nothing to have in flight |
| 1 | `scripts/load_chicago.py` → `chicago.db` | ~½ day | Clone the FARS loader's structure |
| 2 | EDA: base rates, missingness, KA prevalence by segment | ~½ day | Establishes the numbers the model must beat |
| 3 | Baselines — majority, heuristic, logistic | ~½ day | Do not skip; the report needs them |
| 4 | XGBoost + weights, group-temporal split, threshold tuning | ~1 day | |
| 5 | SHAP global + local; SMOTE comparison for the record | ~½ day | The rejected-SMOTE row is a deliverable |
| 6 | Model card + calibration plot | ~½ day | §6.2 |
| 7 | Wrap as `predict_injury_severity` tool | ~½ day | §5.1 |
| 8 | Write AgentEval cases for the tool (§6.3) | ~1 day | Including the causal-refusal `unanswerable` cases |

Roughly **5–6 focused days**. The genuinely novel work is steps 7–8; steps 1–6 are
well-trodden and should not be over-engineered.

**Open item:** spike the Charlotte CDOT ArcGIS REST endpoint (`TCLS_Crashes`) for bulk
export. If it supports it, a Charlotte-specific layer is a strong local-market narrative for
the same effort.

---

## 9. References

**Reviews & methodology**
- [Machine Learning and Deep Learning for Predicting Traffic Crash Injury Severity: A Systematic Review and Meta-Analysis (2014–2025)](https://journalofroadsafety.org/article/156042-machine-learning-and-deep-learning-for-predicting-traffic-crash-injury-severity-a-systematic-review-and-meta-analysis) — Journal of Road Safety
- [A literature review of machine learning algorithms for crash injury severity prediction](https://pubmed.ncbi.nlm.nih.gov/35249605/) — PubMed
- [Recent Advances in Traffic Accident Analysis and Prediction](https://arxiv.org/pdf/2406.13968) — arXiv 2406.13968

**The prediction/causality tradeoff**
- [Mannering, Bhat, Shankar & Abdel-Aty — Big data, traditional data and the tradeoffs between prediction and causality in highway-safety analysis](https://www.caee.utexas.edu/prof/bhat/ABSTRACTS/Prediction_causality.pdf) (PDF) · [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S2213665720300038)
- [Efficient and robust estimation of single-vehicle crash severity: mixed logit with heterogeneity in means and variances](https://www.sciencedirect.com/science/article/abs/pii/S0001457523004931)
- [A comparison of the mixed logit and latent class methods for crash severity analysis](https://www.sciencedirect.com/science/article/abs/pii/S2213665714000244)

**Models, imbalance, interpretability**
- [Crash Injury Severity Prediction Using an Ordinal Classification Machine Learning Approach](https://pmc.ncbi.nlm.nih.gov/articles/PMC8583475/)
- [Traffic accident severity prediction based on an enhanced MSCPO-XGBoost hybrid model](https://www.nature.com/articles/s41598-025-00797-7) — Scientific Reports
- [Machine learning approaches to traffic accident severity prediction: Addressing class imbalance](https://www.sciencedirect.com/science/article/pii/S2666827025001756)
- [Prediction and interpretation of crash severity using machine learning based on imbalanced traffic crash data](https://www.sciencedirect.com/science/article/abs/pii/S0022437525000234) — Journal of Safety Research
- [Predicting and Analyzing Road Traffic Injury Severity Using Boosting-Based Ensemble Learning with SHAP](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8910532/)
- [Deep Learning Model for Crash Injury Severity Analysis Using Shapley Additive Explanation Values](https://doi.org/10.1177/03611981221095087) — Kang & Khattak

**LLMs in crash analysis**
- [CrashSage: A Large Language Model-Centered Framework for Contextual and Interpretable Traffic Crash Analysis](https://arxiv.org/abs/2505.07853)
- [Predicting person-level injury severity using crash narratives](https://arxiv.org/pdf/2509.07845) — arXiv 2509.07845
- [Advanced Crash Causation Analysis for Freeway Safety: An LLM Approach](https://arxiv.org/abs/2505.09949)

**KABCO quality**
- [Comparative analysis of injury identification using KABCO and ISS in linked North Carolina trauma registry and crash data](https://pubmed.ncbi.nlm.nih.gov/38917362/) — Traffic Injury Prevention
- [Accuracy of Injury Severity Ratings on Police Crash Reports](https://journals.sagepub.com/doi/abs/10.3141/2516-09) — Burdett, Li, Bill & Noyce
- [Police-reported pedestrian crash matching and injury severity misclassification by body region](https://www.sciencedirect.com/science/article/abs/pii/S0001457522000094)

**Practice standards**
- [An Introduction to the Highway Safety Manual](https://www.highwaysafetymanual.org/Documents/HSMP-1.pdf) — AASHTO
- [HSM Chapter 4 — Network Screening](https://ftp.granit.unh.edu/submissions/NHDOTOUT/craig/HSM/Chapter%204_NetworkScreening.pdf)
- [CMF Clearinghouse](http://www.cmfclearinghouse.org/resources_spf.cfm)

**Data sources**
- [Chicago Traffic Crashes — Crashes](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if) · [People](https://data.cityofchicago.org/Transportation/Traffic-Crashes-People/u6pd-qa9d) · [Vehicles](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Vehicles/68nd-jvt3)
- [TxDOT CRIS Automated Interface User Guide (Public & Standard Extracts, v29.0)](https://www.txdot.gov/content/dam/docs/division/trf/crash-records/cris-guide.pdf) · [CRIS Query](https://cris.dot.state.tx.us/public/Query/app/home)
- [TxDOT Open Data Portal — AADT Traffic Counts](https://gis-txdot.opendata.arcgis.com/maps/TXDOT::txdot-annual-average-daily-traffic-counts-public) · [Roadway Inventory](https://www.txdot.gov/data-maps/roadway-inventory.html)
- [NHTSA CRSS](https://www.nhtsa.gov/crash-data-systems/crash-report-sampling-system) · [FARS downloads](https://www.nhtsa.gov/file-downloads)
- [HSIS Guidebook — North Carolina (FHWA-HRT-24-108)](https://highways.dot.gov/sites/fhwa.dot.gov/files/FHWA-HRT-24-108.pdf) · [HSIS Overview](https://highways.dot.gov/turner-fairbank-highway-research-center/safety/hsis)
- [Maryland Crash Data Download](https://mdsp.maryland.gov/Pages/Dashboards/CrashDataDownload.aspx) · [Maryland Open Data](https://opendata.maryland.gov/stories/s/Maryland-Crash-Data-Resources/ggbs-m2rv/)
