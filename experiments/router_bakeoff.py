#!/usr/bin/env python3
"""LLM / router bake-off logged to the managed MLflow tracking server.

Each variant -- a model called directly, or a routing policy that picks a model
per request -- becomes one MLflow run carrying the same metric vocabulary, so
the compare views plot cost against quality and latency directly.

Two backends:

  bedrock  real calls through bedrock-runtime Converse in the target account.
  mock     a deterministic local simulator. Costs nothing, needs no model
           access, and is tagged `simulated=true` on every run so a screenshot
           of it can never be mistaken for a measurement of real endpoints.

    python experiments/router_bakeoff.py --backend auto
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.tracking import ConfigRequired, connect  # noqa: E402

EXPERIMENT = "jreq-router-bakeoff"
PROMPTS = Path(__file__).resolve().parent / "data" / "prompts.jsonl"

# Shared parameter vocabulary across every run, sentinel-filled where a knob
# does not apply, so parallel coordinates stays dense.
PARAM_KEYS = ("variant_kind", "policy", "fixed_model", "tier_easy", "tier_hard", "max_tokens")

PRICING_FILE = Path(__file__).resolve().parent / "data" / "pricing.json"


def load_pricing(override: str = "") -> dict:
    """Prices come from a file, not from source, and the file travels with the
    run as an artifact so any figure in a chart can be traced to its input."""
    prices = {k: v for k, v in json.loads(PRICING_FILE.read_text()).items()
              if not k.startswith("_")}
    if override:
        prices.update({k: v for k, v in json.loads(Path(override).read_text()).items()
                       if not k.startswith("_")})
    return prices

# Simulated behaviour for the mock backend: latency in ms and accuracy by tier.
MOCK_PROFILE = {
    "mock-small": {"base_ms": 120, "jitter_ms": 45, "easy": 0.95, "hard": 0.58, "tok_s": 190},
    "mock-medium": {"base_ms": 330, "jitter_ms": 90, "easy": 0.98, "hard": 0.80, "tok_s": 140},
    "mock-large": {"base_ms": 720, "jitter_ms": 160, "easy": 0.99, "hard": 0.93, "tok_s": 95},
}

# Preference order when --models auto is resolved against Bedrock, ascending in
# capability and price: build_variants() treats models[0] as the cheap tier and
# models[-1] as the quality tier, so this order is load-bearing.
BEDROCK_PREFERENCE = [
    "nvidia.nemotron-nano-12b-v2",
    "qwen.qwen3-next-80b-a3b",
    "qwen.qwen3-235b-a22b-2507-v1:0",
    "deepseek.v3.2",
    "moonshotai.kimi-k2.5",
]


@dataclass
class Call:
    prompt_id: str
    tier: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    text: str
    correct: bool
    error: str | None = None


@dataclass
class Variant:
    name: str
    kind: str          # "direct" | "router"
    policy: str        # "none" for direct
    fixed_model: str   # "none" for routers
    tier_map: dict = field(default_factory=dict)


def load_prompts() -> list[dict]:
    if not PROMPTS.exists():
        raise ConfigRequired(f"CONFIG_REQUIRED: prompt set missing at {PROMPTS}")
    rows = [json.loads(line) for line in PROMPTS.read_text().splitlines() if line.strip()]
    if not rows:
        raise ConfigRequired(f"CONFIG_REQUIRED: prompt set at {PROMPTS} is empty")
    return rows


def normalise(text: str) -> str:
    cleaned = text.strip().lower().strip(".,;:!\"'` \n\t")
    return re.sub(r"\s+", " ", cleaned)


def is_correct(response: str, expect: str) -> bool:
    got, want = normalise(response), normalise(expect)
    return got == want or got.endswith(want) and len(got) - len(want) <= 2


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #

def _det_unit(*parts: str) -> float:
    """Deterministic pseudo-random float in [0,1) from the given parts."""
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class MockBackend:
    """Deterministic simulator. Same inputs always produce the same numbers."""

    name = "mock"

    def models(self) -> list[str]:
        return list(MOCK_PROFILE)

    def call(self, model: str, row: dict, max_tokens: int) -> Call:
        profile = MOCK_PROFILE[model]
        jitter = _det_unit("lat", model, row["id"])
        latency = profile["base_ms"] + profile["jitter_ms"] * (jitter * 2 - 1)

        accuracy = profile[row["tier"]]
        correct = _det_unit("acc", model, row["id"]) < accuracy
        text = row["expect"] if correct else f"{row['expect']}x"

        in_tokens = max(8, len(row["prompt"]) // 4)
        out_tokens = max(2, len(text) // 3)
        return Call(row["id"], row["tier"], model, round(latency, 2),
                    in_tokens, out_tokens, text, correct)


class BedrockBackend:
    """Real Converse calls against the target account."""

    name = "bedrock"

    def __init__(self, session, model_ids: list[str]):
        self.client = session.client("bedrock-runtime")
        self._models = model_ids

    def models(self) -> list[str]:
        return list(self._models)

    def call(self, model: str, row: dict, max_tokens: int) -> Call:
        started = time.perf_counter()
        try:
            response = self.client.converse(
                modelId=model,
                messages=[{"role": "user", "content": [{"text": row["prompt"]}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
            )
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            latency = (time.perf_counter() - started) * 1000
            return Call(row["id"], row["tier"], model, round(latency, 2), 0, 0, "", False,
                        error=f"{type(exc).__name__}: {exc}")

        latency = (time.perf_counter() - started) * 1000
        content = response["output"]["message"]["content"]
        text = "".join(part.get("text", "") for part in content)
        usage = response.get("usage", {})
        stop = response.get("stopReason", "")
        error = None
        if not text.strip():
            reasoned = any("reasoningContent" in part for part in content)
            error = ("no text block returned; model spent its budget on reasoning"
                     if reasoned or stop == "max_tokens" else "empty response")
        return Call(
            row["id"], row["tier"], model, round(latency, 2),
            int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0)),
            text, is_correct(text, row["expect"]) if error is None else False,
            error=error,
        )


def discover_bedrock_models(session, requested: str) -> list[str]:
    """Intersect the preference list with what the account can actually call."""
    control = session.client("bedrock")
    available = set()
    try:
        for summary in control.list_foundation_models().get("modelSummaries", []):
            available.add(summary["modelId"])
    except Exception as exc:  # noqa: BLE001
        raise ConfigRequired(f"CONFIG_REQUIRED: cannot list Bedrock models ({exc})") from exc

    try:
        for profile in control.list_inference_profiles().get("inferenceProfileSummaries", []):
            available.add(profile["inferenceProfileId"])
    except Exception:  # noqa: BLE001 - inference profiles are optional
        pass

    if requested != "auto":
        wanted = [m.strip() for m in requested.split(",") if m.strip()]
        missing = [m for m in wanted if m not in available]
        if missing:
            raise ConfigRequired(f"CONFIG_REQUIRED: models not available in this account: {missing}")
        return wanted

    listed = [m for m in BEDROCK_PREFERENCE if m in available]
    runtime = session.client("bedrock-runtime")
    chosen, rejected = [], []
    for model in listed:
        try:
            probe = runtime.converse(
                modelId=model,
                messages=[{"role": "user", "content": [{"text": "Reply with the word ok."}]}],
                inferenceConfig={"maxTokens": 16, "temperature": 0.0},
            )
            text = "".join(p.get("text", "") for p in probe["output"]["message"]["content"])
            if text.strip():
                chosen.append(model)
            else:
                rejected.append(f"{model} (no text block)")
        except Exception as exc:  # noqa: BLE001
            rejected.append(f"{model} ({type(exc).__name__})")
        if len(chosen) == 3:
            break

    if rejected:
        print(f"[router] not usable: {', '.join(rejected)}")
    if len(chosen) < 2:
        raise ConfigRequired(
            "CONFIG_REQUIRED: fewer than two Bedrock models in this account answered a probe "
            f"call (usable: {chosen}). Pass --models explicitly or use --backend mock."
        )
    return chosen


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #

HARD_HINTS = ("round", "percentage", "decimal", "cost", "divided", "frontier", "which", "convert")


def looks_hard(row: dict) -> bool:
    """Difficulty heuristic the routers use. Deliberately imperfect -- that is
    the point of comparing routing policies."""
    prompt = row["prompt"].lower()
    return len(prompt) > 110 or sum(hint in prompt for hint in HARD_HINTS) >= 2


def build_variants(models: list[str]) -> list[Variant]:
    cheap, mid, best = models[0], models[len(models) // 2], models[-1]

    variants = [
        Variant(f"direct::{m}", "direct", "none", m) for m in models
    ]
    variants += [
        Variant("router::cheap-first", "router", "cheap-first", "none",
                {"easy": cheap, "hard": mid}),
        Variant("router::balanced", "router", "balanced", "none",
                {"easy": cheap, "hard": best}),
        Variant("router::quality-first", "router", "quality-first", "none",
                {"easy": mid, "hard": best}),
    ]
    return variants


def pick_model(variant: Variant, row: dict) -> str:
    if variant.kind == "direct":
        return variant.fixed_model
    return variant.tier_map["hard" if looks_hard(row) else "easy"]


# --------------------------------------------------------------------------- #

def score_variant(variant: Variant, backend, rows: list[dict], pricing: dict,
                  max_tokens: int) -> tuple[dict, pd.DataFrame]:
    calls = [backend.call(pick_model(variant, row), row, max_tokens) for row in rows]

    latencies = [c.latency_ms for c in calls]
    ok = [c for c in calls if c.error is None]
    usd = sum(
        c.input_tokens / 1000 * pricing[c.model]["in"] + c.output_tokens / 1000 * pricing[c.model]["out"]
        for c in ok
    )
    total_out = sum(c.output_tokens for c in ok)
    total_seconds = sum(c.latency_ms for c in ok) / 1000 or 1e-9

    metrics = {
        "quality_score": round(sum(c.correct for c in calls) / len(calls), 4),
        "quality_easy": _tier_score(calls, "easy"),
        "quality_hard": _tier_score(calls, "hard"),
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 2),
        "mean_ms": round(statistics.fmean(latencies), 2),
        "tokens_per_sec": round(total_out / total_seconds, 2),
        "input_tokens": sum(c.input_tokens for c in ok),
        "output_tokens": total_out,
        "usd_total": round(usd, 6),
        "usd_per_1k_requests": round(usd / len(calls) * 1000, 4),
        "error_rate": round(sum(c.error is not None for c in calls) / len(calls), 4),
        "n_requests": len(calls),
    }
    frame = pd.DataFrame([c.__dict__ for c in calls])
    return metrics, frame


def _tier_score(calls: list[Call], tier: str) -> float:
    subset = [c for c in calls if c.tier == tier]
    return round(sum(c.correct for c in subset) / len(subset), 4) if subset else 0.0


def frontier_plot(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=140)
    for kind, marker in (("direct", "o"), ("router", "D")):
        subset = frame[frame["variant_kind"] == kind]
        ax.scatter(subset["usd_per_1k_requests"], subset["quality_score"],
                   s=90, marker=marker, label=kind)
    for _, row in frame.iterrows():
        ax.annotate(row["variant"].split("::")[-1],
                    (row["usd_per_1k_requests"], row["quality_score"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xlabel("USD per 1,000 requests")
    ax.set_ylabel("quality score (exact match)")
    ax.set_title("Cost vs quality")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def latency_plot(frame: pd.DataFrame, path: Path) -> None:
    ordered = frame.sort_values("p50_ms")
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=140)
    ax.barh(ordered["variant"], ordered["p50_ms"], label="p50")
    ax.barh(ordered["variant"], ordered["p95_ms"] - ordered["p50_ms"],
            left=ordered["p50_ms"], alpha=0.45, label="p50→p95")
    ax.set_xlabel("latency (ms)")
    ax.set_title("Latency by variant")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=("auto", "bedrock", "mock"), default="auto")
    ap.add_argument("--models", default="auto", help="comma-separated Bedrock model ids, or auto")
    ap.add_argument("--pricing", default="", help=f"JSON file overriding {PRICING_FILE.name}")
    ap.add_argument("--max-tokens", type=int, default=192,
                    help="reasoning models need headroom or they never emit a text block")
    ap.add_argument("--max-error-rate", type=float, default=0.1,
                    help="abort without logging if any variant exceeds this")
    ap.add_argument("--summary-out", default="artifacts/router_runs.csv")
    args = ap.parse_args()

    try:
        target = connect(EXPERIMENT)
        rows = load_prompts()
    except ConfigRequired as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"[router] tracking -> {target.describe()}")

    backend, simulated = _select_backend(args, target)
    if backend is None:
        return 2

    models = backend.models()
    pricing = load_pricing(args.pricing)
    missing_prices = [m for m in models if m not in pricing]
    if missing_prices:
        print(f"CONFIG_REQUIRED: no price configured for {missing_prices}. "
              f"Pass --pricing with a JSON file giving in/out USD per 1k tokens.", file=sys.stderr)
        return 2

    variants = build_variants(models)
    print(f"[router] backend={backend.name} simulated={simulated} models={models} "
          f"variants={len(variants)} prompts={len(rows)}")

    workdir = Path(tempfile.mkdtemp(prefix="jreq-router-"))
    bakeoff_id = time.strftime("%Y%m%d-%H%M%S")
    summary = []

    # Measure everything before writing anything. A backend that is failing
    # every call must not leave a set of plausible-looking runs on the server
    # that only a glance at error_rate would expose as empty.
    measured = []
    for variant in variants:
        print(f"[router] {variant.name} ...", flush=True)
        metrics, calls = score_variant(variant, backend, rows, pricing, args.max_tokens)
        measured.append((variant, metrics, calls))
        print(f"[router]   quality={metrics['quality_score']:.3f} "
              f"p50={metrics['p50_ms']:.0f}ms ${metrics['usd_per_1k_requests']:.3f}/1k "
              f"err={metrics['error_rate']:.2f}")

    worst = max(m["error_rate"] for _, m, _ in measured)
    if worst > args.max_error_rate:
        offenders = [v.name for v, m, _ in measured if m["error_rate"] > args.max_error_rate]
        sample = ""
        for _, _, frame in measured:
            errs = frame["error"].dropna()
            if len(errs):
                sample = str(errs.iloc[0])
                break
        print(f"CONFIG_REQUIRED: error rate {worst:.0%} exceeds the {args.max_error_rate:.0%} "
              f"limit on {len(offenders)} variant(s); nothing was logged.\n"
              f"  first error: {sample}", file=sys.stderr)
        return 2

    for variant, metrics, calls in measured:
        with mlflow.start_run(run_name=variant.name):
            mlflow.set_tags({
                "bakeoff": "router",
                "bakeoff_id": bakeoff_id,
                "backend": backend.name,
                "simulated": str(simulated).lower(),
                "variant_kind": variant.kind,
            })
            mlflow.log_params({
                "variant_kind": variant.kind,
                "policy": variant.policy,
                "fixed_model": variant.fixed_model,
                "tier_easy": variant.tier_map.get("easy", variant.fixed_model),
                "tier_hard": variant.tier_map.get("hard", variant.fixed_model),
                "max_tokens": args.max_tokens,
            })
            mlflow.log_metrics(metrics)

            calls_path = workdir / f"{variant.name.replace('::', '_')}.csv"
            calls.to_csv(calls_path, index=False)
            mlflow.log_artifact(str(calls_path), artifact_path="responses")

            pricing_path = workdir / "pricing.json"
            pricing_path.write_text(json.dumps(pricing, indent=2, sort_keys=True))
            mlflow.log_artifact(str(pricing_path), artifact_path="config")

        summary.append({"variant": variant.name, "variant_kind": variant.kind,
                        "policy": variant.policy, **metrics})

    frame = pd.DataFrame(summary).sort_values("quality_score", ascending=False).reset_index(drop=True)
    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    # One extra run holding the cross-variant plots, so the experiment has a
    # single place to screenshot the frontier from.
    with mlflow.start_run(run_name=f"summary-{bakeoff_id}"):
        mlflow.set_tags({"bakeoff": "router", "bakeoff_id": bakeoff_id, "role": "summary",
                         "simulated": str(simulated).lower(), "backend": backend.name})
        frontier = workdir / "cost_vs_quality.png"
        latency = workdir / "latency_by_variant.png"
        frontier_plot(frame, frontier)
        latency_plot(frame, latency)
        mlflow.log_artifact(str(frontier), artifact_path="plots")
        mlflow.log_artifact(str(latency), artifact_path="plots")
        mlflow.log_artifact(str(out), artifact_path="summary")
        best = frame.iloc[0]
        mlflow.log_metrics({
            "best_quality_score": float(best["quality_score"]),
            "n_variants": len(frame),
        })

    print(f"[router] best quality: {frame.iloc[0]['variant']} ({frame.iloc[0]['quality_score']:.3f})")
    print(f"[router] summary -> {out}")
    return 0


def _select_backend(args, target):
    if args.backend == "mock":
        return MockBackend(), True

    import boto3

    session = boto3.Session(profile_name=target.profile, region_name=target.region)
    try:
        models = discover_bedrock_models(session, args.models)
        return BedrockBackend(session, models), False
    except ConfigRequired as exc:
        if args.backend == "bedrock":
            print(str(exc), file=sys.stderr)
            return None, False
        print(f"[router] bedrock unavailable ({exc}); falling back to the deterministic "
              f"mock backend. Runs will be tagged simulated=true.")
        return MockBackend(), True


if __name__ == "__main__":
    raise SystemExit(main())
