#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------
# Script: setup_github_stg.sh
# You need to install jq and gh (GitHub CLI)
# brew install jq
# brew install gh
# gh auth login
# Purpose:
#  - Set repo secrets: GHA_GCP_AIRUX8_DEPLOYER_SA, MENTERU_TOOLS_DEPLOY_KEY
#  - Create environment: stg
#  - Set environment variables only for stg
# ---------------------------------------------

# --- Config (edit these) ---
ENV_NAME="stg"

# --- Core Environment Variables ---
PROJECT_ID="airux8-opti-logic"
PROJECT_NUM="144706892563"
REGION="asia-northeast1"
DEV_ENV="stg"
OPTIMIZE_ARTIFACT_REPO="airux8-optimize-repo-stg"

# --- Service Account Variables ---
JOB_CLEA_DATA_SCRAPING_SA="job-clea-data-scraping@airux8-opti-logic.iam.gserviceaccount.com"
JOB_ISETAN_DATA_SCRAPING_SA="job-isetan-data-scraping@airux8-opti-logic.iam.gserviceaccount.com"
JOB_TRASS_DATA_LOADER_SA="job-trass-data-loader@airux8-opti-logic.iam.gserviceaccount.com"
FUNC_CHECK_CLEA_FILES_SA="func-check-clea-files@airux8-opti-logic.iam.gserviceaccount.com"
SVC_AIRUX8_OPTIMIZE_SA="svc-airux8-optimize@airux8-opti-logic.iam.gserviceaccount.com"

# --- Storage and Data Path Variables ---
BUCKET_NAME="airux8-opti-logic-stg"
DATA_LOADER_MASTER_DATA_PATH="01_MasterData"
DATA_LOADER_LOADED_DATA_PATH="06_LoadedData"

# --- BigQuery Variables ---
BQ_DATASET_CLEA="Clea"
BQ_DATASET_ISETAN="IsetanMitsukoshi"
BQ_TABLE_AC_CONTROL_RAW="ac_control_raw"
BQ_TABLE_AC_POWER_METER_RAW="ac_power_meter_raw"

# --- Google Drive Variables ---
CLEA_OUT_GDRIVE_FOLDER_ID="1VA9m_cIR5m9j7yfx2t-gnr1vowANRf7O"

# --- Secret Names ---
LOGIN_INFO_SECRET_NAME="AIRUX8_WEB_LOGIN_INFO"

# --- Preconditions ---
command -v gh >/dev/null 2>&1 || { echo "[ERROR] gh is not installed."; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "[ERROR] jq is not installed (brew install jq)."; exit 1; }
command -v gcloud >/dev/null 2>&1 || { echo "[ERROR] gcloud is not installed."; exit 1; }

if ! gh auth status >/dev/null 2>&1; then
  echo "[ERROR] gh is not authenticated. Run: gh auth login"
  exit 1
fi


# Detect owner/repo from git remote
REPO_FULL="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
echo "[INFO] Target repo: $REPO_FULL"

# --- 1) Create repo secrets ---
echo "[INFO] Setting repo secret: GHA_GCP_AIRUX8_DEPLOYER_SA (from GCP)"
DEPLOYER_SA_KEY="$(gcloud secrets versions access latest \
    --secret=GHA_AIRUX8_DEPLOYER_SA \
    --project=$PROJECT_ID)"
gh secret set GHA_GCP_AIRUX8_DEPLOYER_SA --repo "$REPO_FULL" --body "$DEPLOYER_SA_KEY"

echo "[INFO] Setting repo secret: MENTERU_TOOLS_DEPLOY_KEY (from GCP)"
MENTERU_TOOLS_KEY="$(gcloud secrets versions access latest \
    --secret=MENTERU_TOOLS_DEPLOY_KEY \
    --project=menteru-insight-stg)"
gh secret set MENTERU_TOOLS_DEPLOY_KEY --repo "$REPO_FULL" --body "$MENTERU_TOOLS_KEY"


# --- 2) Create environment (stg) if not exists ---
echo "[INFO] Ensuring environment exists: $ENV_NAME"
# GitHub API: create/update environment
gh api -X PUT "repos/${REPO_FULL}/environments/${ENV_NAME}" >/dev/null

# --- 3) Set env variables for STG environment only ---
echo "[INFO] Setting environment variables for: $ENV_NAME"

# Core variables
gh variable set PROJECT_ID --env "$ENV_NAME" --repo "$REPO_FULL" --body "$PROJECT_ID"
gh variable set PROJECT_NUM --env "$ENV_NAME" --repo "$REPO_FULL" --body "$PROJECT_NUM"
gh variable set REGION --env "$ENV_NAME" --repo "$REPO_FULL" --body "$REGION"
gh variable set DEV_ENV --env "$ENV_NAME" --repo "$REPO_FULL" --body "$DEV_ENV"
gh variable set OPTIMIZE_ARTIFACT_REPO --env "$ENV_NAME" --repo "$REPO_FULL" --body "$OPTIMIZE_ARTIFACT_REPO"

# Service Account variables
gh variable set JOB_CLEA_DATA_SCRAPING_SA --env "$ENV_NAME" --repo "$REPO_FULL" --body "$JOB_CLEA_DATA_SCRAPING_SA"
gh variable set JOB_ISETAN_DATA_SCRAPING_SA --env "$ENV_NAME" --repo "$REPO_FULL" --body "$JOB_ISETAN_DATA_SCRAPING_SA"
gh variable set JOB_TRASS_DATA_LOADER_SA --env "$ENV_NAME" --repo "$REPO_FULL" --body "$JOB_TRASS_DATA_LOADER_SA"
gh variable set FUNC_CHECK_CLEA_FILES_SA --env "$ENV_NAME" --repo "$REPO_FULL" --body "$FUNC_CHECK_CLEA_FILES_SA"
gh variable set SVC_AIRUX8_OPTIMIZE_SA --env "$ENV_NAME" --repo "$REPO_FULL" --body "$SVC_AIRUX8_OPTIMIZE_SA"

# Storage and Data Path variables
gh variable set BUCKET_NAME --env "$ENV_NAME" --repo "$REPO_FULL" --body "$BUCKET_NAME"
gh variable set DATA_LOADER_MASTER_DATA_PATH --env "$ENV_NAME" --repo "$REPO_FULL" --body "$DATA_LOADER_MASTER_DATA_PATH"
gh variable set DATA_LOADER_LOADED_DATA_PATH --env "$ENV_NAME" --repo "$REPO_FULL" --body "$DATA_LOADER_LOADED_DATA_PATH"

# BigQuery variables
gh variable set BQ_DATASET_CLEA --env "$ENV_NAME" --repo "$REPO_FULL" --body "$BQ_DATASET_CLEA"
gh variable set BQ_DATASET_ISETAN --env "$ENV_NAME" --repo "$REPO_FULL" --body "$BQ_DATASET_ISETAN"
gh variable set BQ_TABLE_AC_CONTROL_RAW --env "$ENV_NAME" --repo "$REPO_FULL" --body "$BQ_TABLE_AC_CONTROL_RAW"
gh variable set BQ_TABLE_AC_POWER_METER_RAW --env "$ENV_NAME" --repo "$REPO_FULL" --body "$BQ_TABLE_AC_POWER_METER_RAW"

# Google Drive variables
gh variable set CLEA_OUT_GDRIVE_FOLDER_ID --env "$ENV_NAME" --repo "$REPO_FULL" --body "$CLEA_OUT_GDRIVE_FOLDER_ID"

# Secret names
gh variable set LOGIN_INFO_SECRET_NAME --env "$ENV_NAME" --repo "$REPO_FULL" --body "$LOGIN_INFO_SECRET_NAME"


echo ""
echo "[DONE] ✅ Repo secret + stg env vars configured."
echo "       Secret:    GHA_GCP_AIRUX8_DEPLOYER_SA, MENTERU_TOOLS_DEPLOY_KEY (repo-level)"
echo "       Env Vars:  PROJECT_ID, PROJECT_NUM, REGION, DEV_ENV, OPTIMIZE_ARTIFACT_REPO"
echo "                  JOB_CLEA_DATA_SCRAPING_SA, JOB_ISETAN_DATA_SCRAPING_SA, JOB_TRASS_DATA_LOADER_SA"
echo "                  FUNC_CHECK_CLEA_FILES_SA, SVC_AIRUX8_OPTIMIZE_SA"
echo "                  BUCKET_NAME, DATA_LOADER_MASTER_DATA_PATH, DATA_LOADER_LOADED_DATA_PATH"
echo "                  BQ_DATASET_CLEA, BQ_DATASET_ISETAN, BQ_TABLE_AC_CONTROL_RAW, BQ_TABLE_AC_POWER_METER_RAW"
echo "                  CLEA_OUT_GDRIVE_FOLDER_ID, LOGIN_INFO_SECRET_NAME (env: $ENV_NAME)"
