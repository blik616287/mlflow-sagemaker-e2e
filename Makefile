# MLflow on AWS, end to end.
#
#   make check                      read-only: identity, permissions, cost
#   make up EXECUTE=--execute       provision (creates billable resources)
#   make seed                       run both bake-offs
#   make shots                      screenshot the MLflow UI
#   make report                     render artifacts/REPORT.md
#   make down                       stop the server (billing stops, data kept)
#   make start                      restart a stopped server (runs/artifacts intact)
#   make destroy EXECUTE=--execute  remove everything
#   make all EXECUTE=--execute      up -> seed -> shots -> report
#
# Targets that create or delete AWS resources refuse to run without
# EXECUTE=--execute. That gate is deliberate; do not remove it.

SHELL        := /usr/bin/env bash
.SHELLFLAGS  := -Eeuo pipefail -c
.DEFAULT_GOAL := help

AWS_PROFILE ?= spectro
AWS_REGION  ?= us-east-2
EXECUTE     ?=
# Thread budget per model. Unbounded n_jobs on a busy many-core box makes
# fit_seconds measure contention instead of the model; lower this if the
# machine is loaded.
JREQ_N_JOBS ?= 8
VENV        := .venv
PY          := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
TF          := terraform -chdir=terraform

export AWS_PROFILE
export AWS_REGION
export JREQ_N_JOBS

define require_execute
	@if [ "$(EXECUTE)" != "--execute" ]; then \
		echo "refusing to $(1): pass EXECUTE=--execute to confirm"; \
		echo "  make $@ EXECUTE=--execute"; \
		exit 2; \
	fi
endef

.PHONY: help
help:
	@awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' $(MAKEFILE_LIST)

.PHONY: check
check:
	@AWS_PROFILE=$(AWS_PROFILE) AWS_REGION=$(AWS_REGION) ./bin/preflight.sh

$(VENV)/.stamp: experiments/requirements.txt capture/requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -r experiments/requirements.txt -r capture/requirements.txt
	@touch $@

.PHONY: venv
venv: $(VENV)/.stamp
	@echo "venv ready: $(VENV)"

.PHONY: init
init:
	$(TF) init -input=false

.PHONY: plan
plan: init
	$(TF) plan -input=false -out=terraform/plan.out

.PHONY: up
up:
	$(call require_execute,create AWS resources)
	$(TF) init -input=false
	$(TF) apply -input=false -auto-approve
	@echo "waiting for the tracking server to report Created ..."
	@name=$$($(TF) output -raw tracking_server_name); \
	until [ "$$(aws sagemaker describe-mlflow-tracking-server --tracking-server-name $$name \
	          --profile $(AWS_PROFILE) --region $(AWS_REGION) \
	          --query TrackingServerStatus --output text)" = "Created" ]; do \
	  status=$$(aws sagemaker describe-mlflow-tracking-server --tracking-server-name $$name \
	            --profile $(AWS_PROFILE) --region $(AWS_REGION) \
	            --query TrackingServerStatus --output text); \
	  case "$$status" in CreateFailed|DeleteFailed|UpdateFailed) \
	    echo "terminal status $$status"; exit 1;; esac; \
	  echo "  $$status ..."; sleep 30; \
	done; \
	echo "Created."
	@$(TF) output

.PHONY: uri
uri:
	@$(TF) output -raw tracking_server_arn

.PHONY: seed
seed: venv
	MLFLOW_TRACKING_URI=$$($(TF) output -raw tracking_server_arn) $(PY) experiments/tabular_bakeoff.py
	MLFLOW_TRACKING_URI=$$($(TF) output -raw tracking_server_arn) $(PY) experiments/router_bakeoff.py

.PHONY: browsers
browsers: venv
	$(VENV)/bin/playwright install chromium

.PHONY: shots
shots: venv
	MLFLOW_TRACKING_URI=$$($(TF) output -raw tracking_server_arn) $(PY) capture/shoot.py

.PHONY: shots-dry
shots-dry: venv
	MLFLOW_TRACKING_URI=$$($(TF) output -raw tracking_server_arn) $(PY) capture/shoot.py --dry-run

.PHONY: report
report: venv
	MLFLOW_TRACKING_URI=$$($(TF) output -raw tracking_server_arn) $(PY) bin/report.py

.PHONY: ui
ui:
	@aws sagemaker create-presigned-mlflow-tracking-server-url \
	  --tracking-server-name $$($(TF) output -raw tracking_server_name) \
	  --expires-in-seconds 300 --session-expiration-duration-in-seconds 43200 \
	  --profile $(AWS_PROFILE) --region $(AWS_REGION) \
	  --query AuthorizedUrl --output text

.PHONY: down
down:
	aws sagemaker stop-mlflow-tracking-server \
	  --tracking-server-name $$($(TF) output -raw tracking_server_name) \
	  --profile $(AWS_PROFILE) --region $(AWS_REGION)
	@echo "stopped; billing for compute ends, runs and artifacts are kept."

.PHONY: start
start:
	@name=$$($(TF) output -raw tracking_server_name); \
	status=$$(aws sagemaker describe-mlflow-tracking-server --tracking-server-name $$name \
	          --profile $(AWS_PROFILE) --region $(AWS_REGION) \
	          --query TrackingServerStatus --output text); \
	case "$$status" in Created|Started) echo "already running ($$status)."; exit 0;; esac; \
	if [ "$$status" != "Stopped" ]; then \
	  echo "cannot start from status $$status; wait for Stopped."; exit 1; \
	fi; \
	aws sagemaker start-mlflow-tracking-server --tracking-server-name $$name \
	  --profile $(AWS_PROFILE) --region $(AWS_REGION) >/dev/null; \
	echo "starting; compute billing resumes now"; \
	while :; do \
	  s=$$(aws sagemaker describe-mlflow-tracking-server --tracking-server-name $$name \
	       --profile $(AWS_PROFILE) --region $(AWS_REGION) \
	       --query TrackingServerStatus --output text); \
	  case "$$s" in \
	    Created|Started) echo "running ($$s)."; break;; \
	    *Failed) echo "terminal status $$s"; exit 1;; \
	    *) echo "  $$s ..."; sleep 20;; \
	  esac; \
	done

.PHONY: destroy
destroy:
	$(call require_execute,destroy AWS resources)
	@echo "syncing artifacts before destroy ..."
	-aws s3 sync s3://$$($(TF) output -raw artifact_bucket)/mlflow artifacts/s3-mlflow \
	  --profile $(AWS_PROFILE) --region $(AWS_REGION) --quiet
	@echo "emptying the artifact bucket (all versions) ..."
	@bucket=$$($(TF) output -raw artifact_bucket 2>/dev/null || true); \
	if [ -n "$$bucket" ]; then $(PY) bin/empty-bucket.py "$$bucket"; \
	else echo "  no bucket in state"; fi
	$(TF) destroy -input=false -auto-approve -var force_destroy=true
	@echo "verifying nothing is left ..."
	@left=$$(aws sagemaker list-mlflow-tracking-servers --profile $(AWS_PROFILE) --region $(AWS_REGION) \
	         --query 'TrackingServerSummaries[].TrackingServerName' --output text); \
	state=$$($(TF) state list 2>/dev/null | { grep -v '^data\.' || true; } | wc -l); \
	if [ -n "$$left" ] || [ "$$state" -ne 0 ]; then \
	  echo "DESTROY INCOMPLETE: servers=[$$left] terraform-managed-resources=$$state"; \
	  exit 1; \
	fi; \
	echo "clean: no tracking servers, no managed resources in state."

.PHONY: all
all:
	$(call require_execute,run the full cycle)
	$(MAKE) check
	$(MAKE) up EXECUTE=--execute
	$(MAKE) seed
	$(MAKE) shots
	$(MAKE) report
	@echo
	@echo "artifacts/REPORT.md is ready. Tear down with: make destroy EXECUTE=--execute"

.PHONY: e2e
e2e:
	$(call require_execute,run the full cycle under ansible)
	ansible-playbook -i ansible/inventory.ini ansible/site.yml -e execute=true

.PHONY: clean
clean:
	rm -rf $(VENV) artifacts/screenshots/*.png artifacts/*.csv artifacts/REPORT.md artifacts/contact-sheet.png
