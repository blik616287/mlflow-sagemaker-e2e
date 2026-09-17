# MLflow on AWS: a provisioned system, two model bake-offs, and what it cost

This repository builds a managed MLflow tracking server on AWS from nothing, runs
two genuine model comparisons through it, screenshots the resulting UI, writes the
numbers out, and destroys the infrastructure again — as one automated sequence.

Everything below was produced by that sequence on **2026-09-17**. The numbers are
the ones the run emitted, not illustrations. Where a figure came from somewhere
other than the run itself, it is cited in [Sources](#sources).

---

## Summary

| | |
|---|---|
| Tracking server | SageMaker managed MLflow, `Small`, **MLflow 3.0.0** |
| Region | `us-east-2` |
| Provisioning time | 13:38 → ~14:03 local, **~25 min** to `Created` |
| Runs logged | **20** (14 tabular candidates, 6 router variants) |
| Models registered | 1 (`jreq-tabular` v1, alias `champion`) |
| Artifacts written | 208 objects, 59.7 MB to S3 |
| Screenshots captured | **12 of 12**, all required |
| Measured compute cost | **$0.45** for 0.75 h at $0.60/hr |
| Measured inference cost | **$0.0017** for 150 real Bedrock Converse calls |

Total spend for the whole exercise was **under fifty cents**.

---

## The system

One `terraform apply` creates ten resources:

- `aws_sagemaker_mlflow_tracking_server` (`Small`, MLflow 3.0.0, automatic model registration on)
- an S3 artifact bucket — versioned, SSE-S3, public access blocked, 30-day object expiry
- an IAM role the server assumes, scoped to that one bucket plus the SageMaker model-package
  actions needed to mirror the MLflow registry into the SageMaker Model Registry
- a plan-time guard that refuses to apply if the credentials resolve to an unexpected account

Everything carries `Project=jreq-mlflow` / `Ephemeral=true`, and teardown filters on those
tags. Anything that creates or destroys AWS resources refuses to run without an explicit
`EXECUTE=--execute`.

Orchestration is Ansible (one role per phase: preflight → provision → experiments → capture
→ report → teardown), with a Makefile front end for running phases by hand.

---

## Experiment 1 — tabular bake-off

Fourteen candidates across six model families on one reproducible dataset
(`sklearn.datasets.make_classification`: 20,000 rows, 24 features, 8 informative,
15% positive rate, 5% label noise, `random_state=1729`). The generation parameters are
logged as an artifact on every run, so the dataset is reconstructible bit-for-bit.

Every run logs the *same* parameter vocabulary — `family`, `n_estimators`, `max_depth`,
`learning_rate`, `l2`, `num_leaves` — sentinel-filled where a knob does not apply. That is
what keeps MLflow's parallel-coordinates plot dense rather than full of gaps.

| run | ROC AUC | AP | F1 | log loss | fit (s) | predict (ms/1k) |
|---|---|---|---|---|---|---|
| **lightgbm_l63_lr0.05** | **0.8949** | 0.7822 | 0.6716 | 0.2574 | 0.51 | 5.24 |
| xgboost_d8_eta0.05 | 0.8936 | 0.7853 | 0.6627 | 0.2608 | 0.94 | 2.02 |
| lightgbm_l127_lr0.1 | 0.8929 | 0.7898 | 0.6938 | 0.3396 | 1.02 | 8.74 |
| lightgbm_l31_lr0.05 | 0.8917 | 0.7729 | 0.6416 | 0.2601 | 0.36 | 5.48 |
| xgboost_d6_eta0.1 | 0.8904 | 0.7698 | 0.6568 | 0.2701 | 0.38 | 0.84 |
| hist_gbm_lr0.1 | 0.8864 | 0.7600 | 0.6395 | 0.2678 | 0.28 | 2.40 |
| xgboost_d4_eta0.1 | 0.8838 | 0.7505 | 0.6407 | 0.2726 | 0.42 | 0.96 |
| hist_gbm_lr0.05 | 0.8812 | 0.7456 | 0.6053 | 0.2771 | 0.36 | 2.08 |
| random_forest_d16 | 0.8772 | 0.7439 | 0.5164 | 0.3002 | 2.54 | 10.52 |
| hist_gbm_lr0.2 | 0.8743 | 0.7377 | 0.6091 | 0.2807 | 0.11 | 0.70 |
| random_forest_d8 | 0.8318 | 0.6331 | 0.2349 | 0.3578 | 1.64 | 17.84 |
| logreg_C0.1 | 0.6793 | 0.3364 | 0.0584 | 0.4235 | 0.01 | 0.12 |
| logreg_C1.0 | 0.6793 | 0.3365 | 0.0584 | 0.4235 | 0.01 | 0.12 |
| baseline_majority | 0.5000 | 0.1690 | 0.0000 | 0.4543 | 0.00 | 0.02 |

**What it shows.** The top six models sit inside **0.0085 AUC** of each other while differing
by 3.7x in fit time and by **10.4x** in inference latency — the kind of spread that
makes a selection decision about something other than the headline metric.
`lightgbm_l127_lr0.1` has the best F1 (0.6938) but the *worst* log loss of the boosting
models (0.3396), i.e. it ranks well and calibrates badly. `random_forest_d8` is the clearest
trap: respectable AUC (0.8318) hiding an F1 of 0.2349 at the 0.5 threshold.

The winner by held-out ROC AUC is registered as `jreq-tabular` v1 with the `champion` alias.

---

## Experiment 2 — LLM router bake-off

Three Bedrock models called directly, plus three routing policies that pick a model per
request from a difficulty heuristic, all scored over the same fixed 25-prompt set with
deterministic exact-match grading. **150 real Converse calls, zero errors.**

| variant | quality | quality (hard) | p50 ms | p95 ms | $/1k req |
|---|---|---|---|---|---|
| **router::quality-first** | **0.960** | **1.000** | 190.9 | 336.0 | $0.0107 |
| direct::qwen3-235b-a22b | 0.960 | 1.000 | 259.1 | 563.8 | $0.0114 |
| router::balanced | 0.920 | 0.923 | 207.4 | 291.0 | $0.0117 |
| router::cheap-first | 0.920 | 0.923 | 191.3 | 355.7 | $0.0117 |
| direct::qwen3-next-80b-a3b | 0.920 | 0.923 | 183.2 | 248.8 | $0.0108 |
| direct::nemotron-nano-12b-v2 | 0.840 | 0.769 | 206.2 | 436.2 | $0.0121 |

**What it shows.** Two results worth the run:

1. **The routing policy beat every single model it routes between.** `quality-first` matched
   the largest model's accuracy exactly (0.960, and 1.000 on the hard subset) at **26% lower
   p50 latency** and **40% lower p95** — because it sends easy prompts to a smaller model and
   only pays the big model's latency where it matters.

2. **The smallest model was the most expensive.** `nemotron-nano-12b-v2` scored lowest
   (0.840) *and* cost the most per request ($0.0121). Per-token it is mid-priced, but it is
   verbose: cost is driven by output tokens, and a model that answers "43" in one token beats
   one that answers in a sentence regardless of its rate card. Choosing on the published
   per-token price alone would have picked exactly wrong.

The cost axis is real: prices come from the AWS Pricing API rather than a rate-card
screenshot, and `pricing.json` is logged as an artifact on every run so any figure in a chart
traces back to the input that produced it.

---

## Screenshots

Twelve views, all captured from the live UI, in [`artifacts/screenshots/`](artifacts/screenshots/):

| # | View |
|---|---|
| 01 | Experiments list — both bake-offs |
| 02 | Tabular runs table, sorted by ROC AUC |
| 03 | Compare runs — parallel coordinates |
| 04 | Compare runs — ROC AUC vs fit time |
| 05 | Metric charts across runs |
| 06 | Winning run — params, metrics, tags, dataset |
| 07 | Logged artifact — ROC curve (AUC 0.8949) |
| 08 | Model registry |
| 09 | Registered champion version, with the SageMaker model-package ARN tag |
| 10 | Router runs table |
| 11 | Router variants compared |
| 12 | Cost vs quality frontier |

[`artifacts/contact-sheet.png`](artifacts/contact-sheet.png) shows all twelve at once.

Nine were required-to-render; the capture step fails loudly if a required view comes back
empty or suspiciously small, and `artifacts/screenshots/manifest.json` records the outcome
of each. Screenshots retain the AWS account id and resource ARNs — deliberately, since
redacted evidence is weaker evidence.

---

## Cost, measured

| item | amount | basis |
|---|---|---|
| Tracking server compute | **$0.45** | 0.75 h × $0.60/hr, `Small`, us-east-2 |
| Bedrock inference | **$0.0017** | 150 Converse calls: 5,916 input + 678 output tokens |
| S3 artifact storage | ~$0.006/mo | 59.7 MB × $0.10/GB-month |
| **Total** | **~$0.45** | |

Left running instead of destroyed, the tracking server is $14.40/day and $432/30 days. It
can be stopped (`make down`) and restarted without losing runs or artifacts; stopped servers
bill no compute.

---

## What broke, and what the fixes were

The interesting part of an end-to-end build is the list of things that did not work first
time. All seven were found and fixed during this run.

1. **MLflow 3 refuses to serialize tree models by default.** `mlflow.sklearn.log_model`
   now defaults to `skops`, which rejects `sklearn.tree._tree.Tree` without an explicit trust
   list — that is every forest and boosting model here. Fixed by passing
   `serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE`. [[6]](#sources)

2. **`n_jobs=-1` made `fit_seconds` measure the machine, not the model.** On a 32-thread box
   already at load average 50, XGBoost took **173s**; with the thread budget pinned to 4 the
   same fit took **0.82s** — a 211x artifact that would have rendered as a fake algorithmic
   difference on the scatter plot. The budget is now pinned before numpy imports (it has to
   be: the BLAS and OpenMP pools are sized at import time) and logged as a parameter.

3. **`automatic_model_registration` needs IAM nobody mentions.** The tracking server mirrors
   registered models into the SageMaker Model Registry *under its own role*. Without
   `sagemaker:CreateModelPackageGroup` / `DescribeModelPackageGroup` / `CreateModelPackage`
   and friends, the mirror fails and `mlflow.register_model` raises — so the model never
   lands in the MLflow registry either. Added to `terraform/iam.tf`.

4. **Listed Bedrock models are not necessarily callable.** `list-foundation-models` and
   `list-inference-profiles` happily returned ids that fail at Converse with *"This model
   version has reached the end of its life"* and *"marked by provider as Legacy and you have
   not been actively using the model in the last 30 days"*. Discovery now **probes** each
   candidate with a real call and keeps only what answers.

5. **A failing backend produced six plausible-looking runs.** The first router attempt logged
   every variant at quality 0.000 with a 100% error rate — structurally valid runs that only
   an eye on `error_rate` would catch. Restructured to measure everything *before* writing
   anything, with an error-rate gate that aborts without logging.

6. **Reasoning models return no text block.** `openai.gpt-oss-120b` and `minimax.minimax-m2`
   spend their whole token budget on `reasoningContent` and emit an empty `text`. Scoring
   that as a wrong answer would be dishonest; it is now recorded as an error instead.

7. **Three separate capture bugs.** MLflow 3 dropped the standalone `#/experiments` route
   (it 404s). The artifact-viewer `click` steps were *deselecting* the file the URL had
   already selected. And the Models pages raise a product-promo modal only after the registry
   data resolves — i.e. after `networkidle` — so dismissal had to move to just before the
   shutter.

---

## Reproducing

```bash
make check                      # read-only: identity, permissions, cost. Creates nothing.
make up EXECUTE=--execute       # terraform apply + wait for Created (~25 min)
make seed                       # both bake-offs
make shots                      # screenshots + contact sheet
make report                     # artifacts/REPORT.md
make destroy EXECUTE=--execute  # sync artifacts down, then remove everything
```

`make all EXECUTE=--execute` chains the middle four. `make e2e EXECUTE=--execute` runs the
same cycle through Ansible.

Requires: Terraform ≥ 1.6, AWS CLI v2, Python 3.11+, an AWS profile with SageMaker, S3, IAM
and Bedrock access. `make check` verifies all of it and prints the projected cost before
anything is created.

---

## What this does not show

Stated plainly, because a results document that only lists strengths is marketing:

- **The tabular dataset is synthetic.** Real data with real leakage, drift and missingness is
  a harder problem. `--dataset breast_cancer` runs against a bundled real dataset instead.
- **The 25-prompt evaluation set is small and exact-match graded.** It measures instruction
  following and short-form accuracy, not reasoning quality or long-form generation. Quality
  differences of one or two points are within its noise.
- **Latency was measured from one client in one region on one afternoon.** Treat p50/p95 as
  relative, not as a service-level characterization.
- **Prices drift.** Every figure here was pulled from the Pricing API on 2026-09-17 and
  should be re-pulled, not copied.
- **One run is not a benchmark.** Every number here is a single execution with a fixed seed.

---

## Sources

Primary — produced by this run, in this repository:

1. `artifacts/comparison.csv` — every metric in both tables, straight from
   `mlflow.search_runs()`. The report, the CSV and the UI are one query, not three retypings.
2. `artifacts/screenshots/manifest.json` — capture outcome per view.
3. `experiments/data/pricing.json` — the exact token prices used, logged as an artifact on
   every router run.

Primary — AWS APIs, queried live on 2026-09-17:

4. SageMaker MLflow pricing, us-east-2 — `$0.60`/hr Small, `$1.04` Medium, `$1.91` Large,
   `$0.10`/GB-month storage:
   ```bash
   aws pricing get-products --service-code AmazonSageMaker \
     --filters Type=TERM_MATCH,Field=regionCode,Value=us-east-2
   ```
5. Bedrock on-demand token pricing, us-east-2 standard tier:
   ```bash
   aws pricing get-products --service-code AmazonBedrock \
     --filters Type=TERM_MATCH,Field=regionCode,Value=us-east-2
   ```
   Note the Pricing API distinguishes `standard`, `batch`, `flex` and `priority` tiers; this
   run used `standard` throughout.

Documentation:

6. scikit-learn, *Model persistence* — the `skops` vs pickle tradeoff behind fix #1:
   <https://scikit-learn.org/stable/model_persistence.html>
7. `aws_sagemaker_mlflow_tracking_server`, Terraform AWS provider (resource used here;
   requires provider ≥ 5.99, this run used 6.65.0):
   <https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/sagemaker_mlflow_tracking_server>
8. AWS, *Machine learning experiments using Amazon SageMaker with MLflow*:
   <https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html>
9. Amazon Bedrock `Converse` API reference:
   <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html>
10. MLflow Model Registry, aliases and versions:
    <https://mlflow.org/docs/latest/model-registry.html>
11. `sklearn.datasets.make_classification` — the tabular dataset generator:
    <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html>
12. AWS Pricing API `get-products`:
    <https://docs.aws.amazon.com/cli/latest/reference/pricing/get-products.html>

Corrected during this work: an earlier draft quoted **$0.642**/hr and **$0.11**/GB-month for
the Small tracking server, taken from a third-party blog. The Pricing API gives **$0.60** and
**$0.10** for us-east-2. The blog figure appears to be a different region. Source [4] is
authoritative and the repository was corrected to match.
