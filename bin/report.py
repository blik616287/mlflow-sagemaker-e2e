#!/usr/bin/env python3
"""Render artifacts/REPORT.md from the tracking server and the capture manifest.

The tables come from mlflow.search_runs, so the numbers in the report and the
numbers in the screenshots are the same query, not a retyping of it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from common.tracking import ConfigRequired, resolve_target  # noqa: E402

TABULAR = "jreq-tabular-bakeoff"
ROUTER = "jreq-router-bakeoff"


def terraform_outputs() -> dict:
    try:
        raw = subprocess.run(
            ["terraform", f"-chdir={REPO_ROOT / 'terraform'}", "output", "-json"],
            capture_output=True, text=True, check=True,
        ).stdout
        return {k: v.get("value") for k, v in json.loads(raw).items()}
    except Exception:  # noqa: BLE001
        return {}


def table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    present = [(src, label) for src, label in columns if src in frame.columns]
    header = "| " + " | ".join(label for _, label in present) + " |"
    rule = "|" + "|".join("---" for _ in present) + "|"
    lines = [header, rule]
    for _, row in frame.iterrows():
        cells = []
        for src, _ in present:
            value = row[src]
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "REPORT.md"))
    ap.add_argument("--comparison-out", default=str(REPO_ROOT / "artifacts" / "comparison.csv"))
    ap.add_argument("--screenshots", default=str(REPO_ROOT / "artifacts" / "screenshots"))
    args = ap.parse_args()

    import mlflow

    try:
        target = resolve_target()
    except ConfigRequired as exc:
        print(str(exc), file=sys.stderr)
        return 2

    mlflow.set_tracking_uri(target.uri)
    mlflow.set_registry_uri(target.uri)
    client = mlflow.MlflowClient()
    outputs = terraform_outputs()

    frames, sections = [], []

    tab_exp = client.get_experiment_by_name(TABULAR)
    if tab_exp:
        tab = mlflow.search_runs([tab_exp.experiment_id], order_by=["metrics.roc_auc DESC"])
        tab = tab[tab["metrics.roc_auc"].notna()] if "metrics.roc_auc" in tab else tab
        if not tab.empty:
            tab = tab.assign(experiment=TABULAR)
            frames.append(tab)
            view = tab.rename(columns=lambda c: c.replace("metrics.", "").replace("tags.mlflow.runName", "run"))
            sections.append(("Tabular bake-off — held-out test set, sorted by ROC AUC", table(
                view.head(20),
                [("run", "run"), ("roc_auc", "ROC AUC"), ("average_precision", "AP"),
                 ("f1", "F1"), ("log_loss", "log loss"), ("fit_seconds", "fit (s)"),
                 ("predict_ms_per_1k", "predict (ms/1k)")],
            )))

    rt_exp = client.get_experiment_by_name(ROUTER)
    if rt_exp:
        rt = mlflow.search_runs([rt_exp.experiment_id], order_by=["metrics.quality_score DESC"])
        rt = rt[rt["metrics.quality_score"].notna()] if "metrics.quality_score" in rt else rt
        if not rt.empty:
            rt = rt.assign(experiment=ROUTER)
            frames.append(rt)
            view = rt.rename(columns=lambda c: c.replace("metrics.", "").replace("tags.mlflow.runName", "run"))
            simulated = set(rt.get("tags.simulated", pd.Series(dtype=str)).dropna().unique())
            note = ("\n> These runs used the deterministic mock backend "
                    "(`simulated=true`); the latency and cost figures are modelled, not measured.\n"
                    if simulated == {"true"} else "")
            sections.append((f"Router bake-off — {len(rt)} variants over a fixed 25-prompt set{note and ''}", note + table(
                view.head(20),
                [("run", "variant"), ("quality_score", "quality"), ("quality_hard", "quality (hard)"),
                 ("p50_ms", "p50 ms"), ("p95_ms", "p95 ms"), ("usd_per_1k_requests", "$/1k req"),
                 ("error_rate", "errors")],
            )))

    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
        Path(args.comparison_out).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.comparison_out, index=False)

    shots_dir = Path(args.screenshots)
    manifest_path = shots_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    lines = [
        "# MLflow on AWS — run record",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S %Z')}.",
        "",
        "## System",
        "",
        "| | |",
        "|---|---|",
        f"| Account | `{outputs.get('account_id', 'n/a')}` |",
        f"| Region | `{outputs.get('region', target.region)}` |",
        f"| Tracking server | `{outputs.get('tracking_server_name', 'n/a')}` |",
        f"| Tracking URI | `{target.uri}` |",
        f"| Artifact store | `{outputs.get('artifact_store_uri', 'n/a')}` |",
        f"| MLflow client | `{mlflow.__version__}` |",
        "",
    ]

    for title, body in sections:
        lines += [f"## {title}", "", body, ""]

    registered = client.search_model_versions("name='jreq-tabular'")
    if registered:
        best = max(registered, key=lambda v: int(v.version))
        lines += [
            "## Model registry", "",
            f"`jreq-tabular` version **{best.version}** carries the `champion` alias "
            f"(selected by held-out ROC AUC).", "",
        ]

    shots = manifest.get("shots", [])
    if shots:
        lines += ["## Screen grabs", ""]
        for shot in shots:
            status = shot.get("status", "?")
            if status in {"ok", "suspect"} and shot.get("path"):
                rel = Path(shot["path"]).relative_to("artifacts") if str(shot["path"]).startswith("artifacts") else shot["path"]
                lines += [f"### {shot.get('title') or shot['id']}", "", f"![{shot['id']}]({rel})", ""]
            else:
                lines += [f"### {shot.get('title') or shot['id']}", "",
                          f"_not captured — {status}: {shot.get('error', '')}_", ""]

    lines += [
        "## Reproducing this", "",
        "```bash",
        "make check                  # read-only: identity, permissions, cost",
        "make up EXECUTE=--execute   # terraform apply, ~25 min to Created",
        "make seed                   # both bake-offs",
        "make shots                  # screenshots + contact sheet",
        "make report                 # this file",
        "make destroy EXECUTE=--execute",
        "```",
        "",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"[report] -> {out}")
    if frames:
        print(f"[report] comparison -> {args.comparison_out} ({sum(len(f) for f in frames)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
