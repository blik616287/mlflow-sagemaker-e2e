#!/usr/bin/env bash
# Read-only preflight. Creates nothing, changes nothing, and is safe to run at
# any time. Exits non-zero when the environment cannot support a run.
set -Eeuo pipefail

PROFILE="${AWS_PROFILE:-spectro}"
REGION="${AWS_REGION:-us-east-2}"
SIZE="${TRACKING_SERVER_SIZE:-Small}"

fail=0
say()  { printf '  %-22s %s\n' "$1" "$2"; }
bad()  { printf '  %-22s %s\n' "$1" "$2"; fail=1; }

echo "== binaries =="
for bin in terraform aws python3 kubectl montage; do
  if command -v "$bin" >/dev/null 2>&1; then
    say "$bin" "$(command -v "$bin")"
  elif [ "$bin" = "kubectl" ] || [ "$bin" = "montage" ]; then
    say "$bin" "absent (optional)"
  else
    bad "$bin" "MISSING (CONFIG_REQUIRED)"
  fi
done

echo
echo "== aws identity =="
if ident=$(aws sts get-caller-identity --profile "$PROFILE" --output json 2>&1); then
  account=$(printf '%s' "$ident" | python3 -c 'import json,sys; print(json.load(sys.stdin)["Account"])')
  arn=$(printf '%s' "$ident" | python3 -c 'import json,sys; print(json.load(sys.stdin)["Arn"])')
  say "profile" "$PROFILE"
  say "account" "$account"
  say "principal" "$arn"
  say "region" "$REGION"
else
  bad "sts" "cannot resolve credentials for profile '$PROFILE'"
  printf '%s\n' "$ident" | sed 's/^/      /'
fi

echo
echo "== sagemaker mlflow reachability =="
if servers=$(aws sagemaker list-mlflow-tracking-servers --profile "$PROFILE" --region "$REGION" --output json 2>&1); then
  count=$(printf '%s' "$servers" | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("TrackingServerSummaries",[])))')
  say "list call" "ok"
  say "existing servers" "$count in $REGION"
  if [ "$count" != "0" ]; then
    printf '%s' "$servers" | python3 -c '
import json,sys
for s in json.load(sys.stdin).get("TrackingServerSummaries", []):
    print(f"      - {s[\"TrackingServerName\"]}  {s.get(\"TrackingServerStatus\")}  {s.get(\"CreationTime\")}")'
  fi
else
  bad "list call" "denied or unavailable"
  printf '%s\n' "$servers" | sed 's/^/      /'
fi

echo
echo "== cost =="
case "$SIZE" in
  Small)  rate=0.60 ;;
  Medium) rate=1.04 ;;
  Large)  rate=1.91 ;;
  *)      rate=0.60 ;;
esac
say "size" "$SIZE"
say "rate" "\$$rate/hr while running, \$0 while stopped"
say "typical pass" "$(python3 -c "print(f'~\${$rate*1.5:.2f} for a 90-minute up/capture/down cycle')")"
say "if left running" "$(python3 -c "print(f'\${$rate*24:.2f}/day, \${$rate*24*30:.2f}/30 days')")"

echo
if [ "$fail" -ne 0 ]; then
  echo "PREFLIGHT FAILED (CONFIG_REQUIRED above)"
  exit 1
fi
echo "PREFLIGHT OK"
