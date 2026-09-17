#!/usr/bin/env python3
"""Tabular model bake-off logged to the managed MLflow tracking server.

Trains a spread of classifiers on one reproducible dataset, logs every run with
the same parameter vocabulary so MLflow's compare views (parallel coordinates,
scatter, metric charts) have dense axes, and registers the best model.

    python experiments/tabular_bakeoff.py --dataset synthetic
"""

from __future__ import annotations

import argparse
import json
import os

# Pinned before numpy/sklearn/xgboost import, or the BLAS and OpenMP pools are
# already sized by then. Unbounded n_jobs on a busy 32-thread hybrid-core box
# makes fit_seconds a measurement of thread contention rather than of the model.
N_JOBS = int(os.environ.get("JREQ_N_JOBS", "8"))
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, str(N_JOBS))
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.tracking import ConfigRequired, connect, timed  # noqa: E402

EXPERIMENT = "jreq-tabular-bakeoff"
REGISTERED_MODEL = "jreq-tabular"
SEED = 1729

# Every run logs this exact parameter vocabulary. Families that do not use a
# knob log the sentinel rather than omitting the key, which is what keeps the
# parallel-coordinates plot dense instead of full of gaps.
PARAM_KEYS = ("family", "n_estimators", "max_depth", "learning_rate", "l2", "num_leaves")
SENTINEL = 0


def build_dataset(kind: str):
    """Return X_train, X_test, y_train, y_test, feature_names, provenance."""
    if kind == "synthetic":
        # A deliberately imperfect, credit-risk-shaped problem: informative and
        # redundant features, class imbalance, and label noise, so the models
        # actually separate instead of all landing at AUC 0.99.
        X, y = make_classification(
            n_samples=20_000,
            n_features=24,
            n_informative=8,
            n_redundant=6,
            n_repeated=0,
            n_clusters_per_class=3,
            weights=[0.85, 0.15],
            flip_y=0.05,
            class_sep=0.8,
            random_state=SEED,
        )
        names = [f"f{i:02d}" for i in range(X.shape[1])]
        provenance = {
            "source": "sklearn.datasets.make_classification",
            "n_samples": 20_000,
            "n_features": 24,
            "n_informative": 8,
            "positive_rate": 0.15,
            "flip_y": 0.05,
            "random_state": SEED,
            "note": "synthetic by design; parameters recorded so the set is reproducible bit-for-bit",
        }
    elif kind == "breast_cancer":
        bunch = load_breast_cancer()
        X, y = bunch.data, bunch.target
        names = list(bunch.feature_names)
        provenance = {
            "source": "sklearn.datasets.load_breast_cancer",
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "random_state": SEED,
        }
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"unknown dataset {kind!r}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    return X_train, X_test, y_train, y_test, names, provenance


def model_grid():
    """(run_name, params, estimator_factory) tuples."""
    grid = []

    grid.append(
        (
            "baseline_majority",
            {"family": "baseline", "n_estimators": SENTINEL, "max_depth": SENTINEL,
             "learning_rate": SENTINEL, "l2": SENTINEL, "num_leaves": SENTINEL},
            lambda: DummyClassifier(strategy="prior"),
        )
    )

    for c in (0.1, 1.0):
        grid.append(
            (
                f"logreg_C{c}",
                {"family": "logreg", "n_estimators": SENTINEL, "max_depth": SENTINEL,
                 "learning_rate": SENTINEL, "l2": round(1.0 / c, 4), "num_leaves": SENTINEL},
                lambda c=c: Pipeline(
                    [("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=c, max_iter=2000, random_state=SEED))]
                ),
            )
        )

    for depth in (8, 16):
        grid.append(
            (
                f"random_forest_d{depth}",
                {"family": "random_forest", "n_estimators": 300, "max_depth": depth,
                 "learning_rate": SENTINEL, "l2": SENTINEL, "num_leaves": SENTINEL},
                lambda depth=depth: RandomForestClassifier(
                    n_estimators=300, max_depth=depth, n_jobs=N_JOBS, random_state=SEED
                ),
            )
        )

    for lr in (0.05, 0.10, 0.20):
        grid.append(
            (
                f"hist_gbm_lr{lr}",
                {"family": "hist_gbm", "n_estimators": 300, "max_depth": 6,
                 "learning_rate": lr, "l2": 1.0, "num_leaves": 31},
                lambda lr=lr: HistGradientBoostingClassifier(
                    max_iter=300, max_depth=6, learning_rate=lr, l2_regularization=1.0,
                    random_state=SEED
                ),
            )
        )

    from xgboost import XGBClassifier

    for depth, eta in ((4, 0.10), (6, 0.10), (8, 0.05)):
        grid.append(
            (
                f"xgboost_d{depth}_eta{eta}",
                {"family": "xgboost", "n_estimators": 400, "max_depth": depth,
                 "learning_rate": eta, "l2": 1.0, "num_leaves": SENTINEL},
                lambda depth=depth, eta=eta: XGBClassifier(
                    n_estimators=400, max_depth=depth, learning_rate=eta, reg_lambda=1.0,
                    subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                    eval_metric="logloss", n_jobs=N_JOBS, random_state=SEED,
                ),
            )
        )

    from lightgbm import LGBMClassifier

    for leaves, lr in ((31, 0.05), (63, 0.05), (127, 0.10)):
        grid.append(
            (
                f"lightgbm_l{leaves}_lr{lr}",
                {"family": "lightgbm", "n_estimators": 400, "max_depth": -1,
                 "learning_rate": lr, "l2": 1.0, "num_leaves": leaves},
                lambda leaves=leaves, lr=lr: LGBMClassifier(
                    n_estimators=400, num_leaves=leaves, learning_rate=lr, reg_lambda=1.0,
                    subsample=0.9, colsample_bytree=0.9, n_jobs=N_JOBS, random_state=SEED,
                    verbose=-1,
                ),
            )
        )

    return grid


def positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raw = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-raw))


def plot_roc(y_true, scores, title, path):
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=140)
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#888")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC — {title}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pr(y_true, scores, title, path):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=140)
    ax.plot(recall, precision, linewidth=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–recall — {title}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=140)
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(f"Confusion — {title}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_importance(model, names, title, path) -> bool:
    est = model.named_steps["clf"] if isinstance(model, Pipeline) else model
    if hasattr(est, "feature_importances_"):
        values = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        values = np.abs(np.asarray(est.coef_, dtype=float)).ravel()
    else:
        return False

    order = np.argsort(values)[::-1][:15][::-1]
    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=140)
    ax.barh([names[i] for i in order], values[order])
    ax.set_xlabel("importance")
    ax.set_title(f"Top features — {title}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def run_one(name, params, factory, data, names, workdir: Path,
            bakeoff_id: str, dataset: str, provenance_path: Path) -> dict:
    X_train, X_test, y_train, y_test = data
    timings: dict[str, float] = {}

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({
            "bakeoff": "tabular",
            "bakeoff_id": bakeoff_id,
            "family": params["family"],
            "seed": str(SEED),
        })
        mlflow.log_params({k: params[k] for k in PARAM_KEYS})
        mlflow.log_param("dataset", dataset)
        mlflow.log_param("n_train", int(X_train.shape[0]))
        mlflow.log_param("n_features", int(X_train.shape[1]))
        mlflow.log_param("n_jobs", N_JOBS)
        mlflow.log_artifact(str(provenance_path), artifact_path="dataset")

        model = factory()
        with timed(timings, "fit_seconds"):
            model.fit(X_train, y_train)

        with timed(timings, "predict_seconds"):
            scores = positive_scores(model, X_test)
        preds = (scores >= 0.5).astype(int)

        metrics = {
            "roc_auc": float(roc_auc_score(y_test, scores)),
            "average_precision": float(average_precision_score(y_test, scores)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "accuracy": float(accuracy_score(y_test, preds)),
            "log_loss": float(log_loss(y_test, np.clip(scores, 1e-6, 1 - 1e-6))),
            "brier": float(brier_score_loss(y_test, scores)),
            "fit_seconds": timings["fit_seconds"],
            "predict_ms_per_1k": round(timings["predict_seconds"] / len(y_test) * 1e6, 4),
        }
        mlflow.log_metrics(metrics)

        run_dir = workdir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        plot_roc(y_test, scores, name, run_dir / "roc_curve.png")
        plot_pr(y_test, scores, name, run_dir / "pr_curve.png")
        plot_confusion(y_test, preds, name, run_dir / "confusion_matrix.png")
        had_importance = plot_importance(model, names, name, run_dir / "feature_importance.png")
        mlflow.log_artifacts(str(run_dir), artifact_path="plots")
        mlflow.set_tag("has_feature_importance", str(had_importance).lower())

        sample = pd.DataFrame({"y_true": y_test[:2000], "score": scores[:2000], "y_pred": preds[:2000]})
        sample_path = run_dir / "predictions_sample.csv"
        sample.to_csv(sample_path, index=False)
        mlflow.log_artifact(str(sample_path), artifact_path="predictions")

        signature = mlflow.models.infer_signature(X_test[:50], scores[:50])
        # cloudpickle rather than the skops default: skops refuses to serialise
        # sklearn.tree._tree.Tree without an explicit trust list, which every
        # forest and boosting model here contains.
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_test[:5],
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        mlflow.log_metric("model_size_bytes", _artifact_bytes(model))

        return {"run_name": name, "run_id": run.info.run_id, "model_uri": info.model_uri, **metrics}


def _artifact_bytes(model) -> int:
    import pickle

    with tempfile.NamedTemporaryFile(delete=True) as fh:
        pickle.dump(model, fh)
        fh.flush()
        return os.path.getsize(fh.name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("synthetic", "breast_cancer"), default="synthetic")
    ap.add_argument("--summary-out", default="artifacts/tabular_runs.csv")
    args = ap.parse_args()

    try:
        target = connect(EXPERIMENT)
    except ConfigRequired as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"[tabular] tracking -> {target.describe()}")

    X_train, X_test, y_train, y_test, names, provenance = build_dataset(args.dataset)
    grid = model_grid()
    print(f"[tabular] dataset={args.dataset} train={X_train.shape} test={X_test.shape} runs={len(grid)}")

    workdir = Path(tempfile.mkdtemp(prefix="jreq-tabular-"))
    started = time.time()
    rows = []

    bakeoff_id = time.strftime("%Y%m%d-%H%M%S")
    prov_path = workdir / "dataset.json"
    prov_path.write_text(json.dumps(provenance, indent=2))

    for name, params, factory in grid:
        print(f"[tabular] {name} ...", flush=True)
        rows.append(
            run_one(
                name, params, factory,
                (X_train, X_test, y_train, y_test),
                names, workdir, bakeoff_id, args.dataset, prov_path,
            )
        )
        print(f"[tabular]   auc={rows[-1]['roc_auc']:.4f} f1={rows[-1]['f1']:.4f} "
              f"fit={rows[-1]['fit_seconds']:.2f}s")

    print(f"[tabular] {len(rows)} runs in {time.time() - started:.1f}s")

    frame = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    best = frame.iloc[0]
    print(f"[tabular] best: {best['run_name']} auc={best['roc_auc']:.4f}")

    version = mlflow.register_model(best["model_uri"], REGISTERED_MODEL)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", version.version)
    client.set_model_version_tag(REGISTERED_MODEL, version.version, "selected_by", "roc_auc")
    client.set_model_version_tag(REGISTERED_MODEL, version.version, "roc_auc", f"{best['roc_auc']:.6f}")
    client.update_registered_model(
        REGISTERED_MODEL,
        description=f"Best of {len(grid)} candidates on the {args.dataset} dataset, selected by held-out ROC AUC.",
    )
    print(f"[tabular] registered {REGISTERED_MODEL} v{version.version} as @champion")
    print(f"[tabular] summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
