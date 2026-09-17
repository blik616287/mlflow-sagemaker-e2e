#!/usr/bin/env python3
"""Screenshot the managed MLflow UI.

Resolves the ids of the runs that were just logged, mints a presigned UI URL,
drives headless chromium through the views listed in shots.yaml, and writes a
manifest recording what was captured and what was not.

    python capture/shoot.py --out artifacts/screenshots
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from common.tracking import ConfigRequired, resolve_target  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TABULAR_EXPERIMENT = "jreq-tabular-bakeoff"
ROUTER_EXPERIMENT = "jreq-router-bakeoff"
REGISTERED_MODEL = "jreq-tabular"


def json_list(values) -> str:
    """MLflow compare-run routes take a JSON array in a query parameter."""
    return quote(json.dumps(list(values)), safe="")


def resolve_context(target) -> dict:
    import mlflow

    mlflow.set_tracking_uri(target.uri)
    mlflow.set_registry_uri(target.uri)
    client = mlflow.MlflowClient()

    variants = []
    tabular = client.get_experiment_by_name(TABULAR_EXPERIMENT)
    router = client.get_experiment_by_name(ROUTER_EXPERIMENT)
    if tabular is None:
        raise ConfigRequired(
            f"CONFIG_REQUIRED: experiment {TABULAR_EXPERIMENT} does not exist yet. "
            "Run the experiments step before capturing."
        )

    tab_runs = client.search_runs(
        [tabular.experiment_id], order_by=["metrics.roc_auc DESC"], max_results=200
    )
    tab_runs = [r for r in tab_runs if "roc_auc" in r.data.metrics]
    if not tab_runs:
        raise ConfigRequired(f"CONFIG_REQUIRED: {TABULAR_EXPERIMENT} has no runs with a roc_auc metric.")

    ctx = {
        "tabular_exp_id": tabular.experiment_id,
        "tabular_exp_list": json_list([tabular.experiment_id]),
        "tabular_best_run_id": tab_runs[0].info.run_id,
        "tabular_compare_runs": json_list([r.info.run_id for r in tab_runs[:8]]),
        "registered_model": REGISTERED_MODEL,
        "registered_version": "1",
    }

    versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    if versions:
        ctx["registered_version"] = str(max(int(v.version) for v in versions))

    if router is not None:
        rt_runs = client.search_runs(
            [router.experiment_id], order_by=["metrics.quality_score DESC"], max_results=200
        )
        variants = [r for r in rt_runs if r.data.tags.get("role") != "summary"]
        summary = next((r for r in rt_runs if r.data.tags.get("role") == "summary"), None)
        ctx.update({
            "router_exp_id": router.experiment_id,
            "router_exp_list": json_list([router.experiment_id]),
            "router_compare_runs": json_list([r.info.run_id for r in variants[:8]]),
            "router_summary_run_id": summary.info.run_id if summary else (
                variants[0].info.run_id if variants else ""
            ),
        })

    ctx["_meta"] = {
        "tracking_uri": target.uri,
        "region": target.region,
        "tabular_run_count": len(tab_runs),
        "router_run_count": len(variants) if router is not None else 0,
    }
    return ctx


def presigned_url(target, server_name: str, expires_in: int, session_seconds: int) -> str:
    import boto3

    session = boto3.Session(profile_name=target.profile, region_name=target.region)
    sm = session.client("sagemaker")
    return sm.create_presigned_mlflow_tracking_server_url(
        TrackingServerName=server_name,
        ExpiresInSeconds=expires_in,
        SessionExpirationDurationInSeconds=session_seconds,
    )["AuthorizedUrl"]


def server_name_from(target) -> str:
    # arn:aws:sagemaker:<region>:<acct>:mlflow-tracking-server/<name>
    if "/" not in target.uri:
        raise ConfigRequired(
            f"CONFIG_REQUIRED: tracking uri {target.uri!r} is not a tracking-server ARN, "
            "so a presigned UI URL cannot be minted."
        )
    return target.uri.rsplit("/", 1)[-1]


REDACT_JS = """(token) => {
  if (!token) return 0;
  let n = 0;
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) {
    if (node.nodeValue && node.nodeValue.includes(token)) {
      node.nodeValue = node.nodeValue.split(token).join('\\u2039account\\u203a');
      n++;
    }
  }
  for (const el of document.querySelectorAll('[title],[aria-label]')) {
    for (const attr of ['title', 'aria-label']) {
      const v = el.getAttribute(attr);
      if (v && v.includes(token)) {
        el.setAttribute(attr, v.split(token).join('\\u2039account\\u203a'));
        n++;
      }
    }
  }
  return n;
}"""


def redact_page(page, token: str, log: list[str]) -> None:
    """Mask the AWS account number in the rendered DOM before the shutter.

    These screenshots are published. Painting boxes over pixels afterwards is
    fragile and lossy; rewriting the text nodes is exact and reproducible.
    """
    if not token:
        return
    try:
        hits = page.evaluate(REDACT_JS, token)
        if hits:
            log.append(f"redacted account id in {hits} node(s)")
    except Exception as exc:  # noqa: BLE001
        log.append(f"redaction failed ({type(exc).__name__})")


def dismiss_overlays(page, log: list[str]) -> None:
    """MLflow 3 pops product-promo modals over the Models pages. They are not
    part of the evidence, so they are closed before anything is captured."""
    for _ in range(3):
        if page.get_by_role("dialog").count() == 0:
            return
        try:
            page.locator('[aria-label="Close"]').first.click(timeout=2000)
        except Exception:  # noqa: BLE001
            page.keyboard.press("Escape")
        page.wait_for_timeout(800)
        log.append("dismissed an overlay dialog")


def apply_selects(page, selects, log: list[str]) -> None:
    """Drive the axis dropdowns. MLflow picks defaults alphabetically, which on
    the scatter tab lands on a categorical param and produces a single-column
    plot -- rendered, but useless as evidence."""
    for entry in selects or []:
        index, option = entry["index"], entry["option"]
        optional = entry.get("optional", False)
        try:
            page.get_by_role("combobox").nth(index).click(timeout=6000)
            page.wait_for_timeout(700)
            page.get_by_role("option", name=option, exact=True).first.click(timeout=6000)
            page.wait_for_timeout(1000)
            log.append(f"combobox[{index}] = {option!r}")
        except Exception as exc:  # noqa: BLE001
            if not optional:
                raise
            log.append(f"combobox[{index}] -> {option!r} skipped ({type(exc).__name__})")


def apply_clicks(page, clicks, log: list[str]) -> None:
    for entry in clicks or []:
        label, optional = entry["text"], entry.get("optional", False)
        try:
            page.get_by_text(label, exact=False).first.click(timeout=4000)
            log.append(f"clicked {label!r}")
            page.wait_for_timeout(900)
        except Exception as exc:  # noqa: BLE001
            if not optional:
                raise
            log.append(f"skipped {label!r} ({type(exc).__name__})")


def capture(args) -> int:
    try:
        target = resolve_target()
        plan = yaml.safe_load(Path(args.plan).read_text())
        ctx = resolve_context(target)
    except ConfigRequired as exc:
        print(str(exc), file=sys.stderr)
        return 2

    defaults = plan.get("defaults", {})
    shots = plan["shots"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for shot in shots:
            try:
                print(f"{shot['id']:38} {shot['route'].format(**ctx)}")
            except KeyError as missing:
                print(f"{shot['id']:38} UNRESOLVED placeholder {missing}")
        return 0

    from playwright.sync_api import sync_playwright

    url = presigned_url(target, server_name_from(target), args.expires_in, args.session_seconds)
    print(f"[capture] presigned URL minted (expires in {args.expires_in}s)")

    # arn:aws:sagemaker:<region>:<account>:mlflow-tracking-server/<name>
    parts = target.uri.split(":")
    redact_token = "" if args.no_redact else (parts[4] if len(parts) > 5 else "")
    if redact_token:
        print(f"[capture] redacting account id {redact_token[:4]}******** from every view")

    viewport = defaults.get("viewport", {"width": 1920, "height": 1080})
    manifest, failures = [], []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        context = browser.new_context(viewport=viewport, device_scale_factor=args.scale)
        page = context.new_page()
        page.set_default_timeout(args.timeout_ms)

        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)
        base = page.url.split("#")[0]
        print(f"[capture] session established at {base}")

        for shot in shots:
            record = {"id": shot["id"], "title": shot.get("title", ""),
                      "required": shot.get("required", False), "notes": []}
            try:
                route = shot["route"].format(**ctx)
            except KeyError as missing:
                record.update(status="unresolved", error=f"missing context key {missing}")
                manifest.append(record)
                if record["required"]:
                    failures.append(record)
                print(f"[capture] {shot['id']}: UNRESOLVED {missing}")
                continue

            target_url = base + route
            # The manifest is published. The route is the useful part; the
            # host identifies the account as surely as the account id does.
            record["url"] = "<tracking-server>/" + route
            try:
                page.goto(target_url, wait_until="domcontentloaded")
                page.wait_for_load_state("networkidle")
                dismiss_overlays(page, record["notes"])
                for selector in shot.get("wait_for", []):
                    page.wait_for_selector(selector.format(**ctx), timeout=args.timeout_ms)
                apply_clicks(page, shot.get("click"), record["notes"])
                apply_selects(page, shot.get("select"), record["notes"])
                page.wait_for_timeout(shot.get("settle_ms", defaults.get("settle_ms", 1200)))
                # Again right before the shutter: the Models pages raise their
                # promo modal only once the registry data resolves, which is
                # after networkidle.
                dismiss_overlays(page, record["notes"])
                redact_page(page, redact_token, record["notes"])

                path = out_dir / f"{shot['id']}.png"
                page.screenshot(path=str(path),
                                full_page=shot.get("full_page", defaults.get("full_page", False)))
                size = path.stat().st_size
                record.update(status="ok", path=str(path.relative_to(REPO_ROOT)), bytes=size)
                if size < args.min_bytes:
                    record["status"] = "suspect"
                    record["notes"].append(f"only {size} bytes; the view may not have rendered")
                    if record["required"]:
                        failures.append(record)
                print(f"[capture] {shot['id']}: {record['status']} ({size:,} bytes)")
            except Exception as exc:  # noqa: BLE001
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                print(f"[capture] {shot['id']}: FAILED {exc}", file=sys.stderr)
                if record["required"]:
                    failures.append(record)

            manifest.append(record)

        context.close()
        browser.close()

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tracking_uri": target.uri.replace(redact_token, "<account>") if redact_token else target.uri,
        "region": target.region,
        "context": {k: v for k, v in ctx.items() if not k.startswith("_")},
        "shots": manifest,
    }, indent=2))
    print(f"[capture] manifest -> {manifest_path}")

    contact_sheet(out_dir, args.contact_sheet)

    if failures:
        print(f"[capture] {len(failures)} required shot(s) did not render:", file=sys.stderr)
        for record in failures:
            print(f"  - {record['id']}: {record.get('error') or record['notes']}", file=sys.stderr)
        return 1
    return 0


def contact_sheet(out_dir: Path, dest: str) -> None:
    images = sorted(out_dir.glob("*.png"))
    if not images:
        return
    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["montage", *[str(p) for p in images], "-tile", "3x", "-geometry", "640x+8+8",
             "-background", "#101014", str(target)],
            check=True, capture_output=True,
        )
        print(f"[capture] contact sheet -> {target}")
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[capture] contact sheet skipped ({exc})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default=str(REPO_ROOT / "capture" / "shots.yaml"))
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "screenshots"))
    ap.add_argument("--contact-sheet", default=str(REPO_ROOT / "artifacts" / "contact-sheet.png"))
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--timeout-ms", type=int, default=30000)
    ap.add_argument("--expires-in", type=int, default=300)
    ap.add_argument("--session-seconds", type=int, default=43200)
    ap.add_argument("--min-bytes", type=int, default=50_000)
    ap.add_argument("--no-redact", action="store_true",
                    help="leave the AWS account id visible (default is to mask it)")
    ap.add_argument("--headed", action="store_true", help="watch the run in a real window")
    ap.add_argument("--dry-run", action="store_true", help="resolve and print URLs, no browser")
    return capture(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
