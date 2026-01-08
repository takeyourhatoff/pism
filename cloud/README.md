# PISM Cloud (AWS Batch)

Minimal, production-shaped tooling to run PISM ensembles as single-node AWS Batch jobs.

## Repository layout

- `cloud/pism_cloud/` - CLI and shared helpers.
- `cloud/infra/cloudformation/` - AWS Batch, IAM, DynamoDB, CloudWatch infra.
- `cloud/dashboard/` - FastAPI dashboard.
- `cloud/examples/` - Example `config.yml`.

## Run this in 30 minutes

### 1) Prerequisites

- AWS account + configured credentials (`aws sts get-caller-identity` should work).
- Terraform 1.4+.
- Python 3.9+.
- Network access to pull container images from GHCR (or copy to ECR).

### 2) Create S3 buckets

```
aws s3 mb s3://pism-inputs
aws s3 mb s3://pism-results
```

Upload your inputs to:

```
s3://pism-inputs/<run-id>/
```

### 3) Deploy infrastructure

```
aws cloudformation deploy \
  --stack-name pism-batch \
  --template-file cloud/infra/cloudformation/pism-batch.yml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    SubnetIds=subnet-abc,subnet-def \
    SecurityGroupIds=sg-123 \
    InputBucketArn=arn:aws:s3:::pism-inputs \
    OutputBucketArn=arn:aws:s3:::pism-results
```

The stack creates Batch queues, job definitions, a log group, and the DynamoDB table.

### 4) Install the CLI

```
python -m venv .venv
source .venv/bin/activate
pip install -e cloud
```

### 5) Submit a run

```
pism-cloud submit cloud/examples/config.yml
```

### 6) Watch status

```
pism-cloud status biis-test-001
```

### 7) Launch the dashboard

```
uvicorn dashboard.app:app --app-dir cloud --port 8080
```

Open `http://localhost:8080`.

## Configuration

Placeholders inside `pism.args` are expanded per job:

- `{{INPUT_DIR}}` -> `/workspace/input`
- `{{OUTPUT_DIR}}` -> `/workspace/output`
- `{{RUN_ID}}`
- `{{MEMBER_ID}}`

Jobs download inputs with `aws s3 sync`, run PISM, then sync outputs to
`io.output_s3/<member-id>/`.

If your container uses `pism` instead of `pismr`, set `pism.executable: pism`.

## Cost policy

- Static on-demand rates live in `cloud/pism_cloud/catalog.py`.
- Spot discount defaults to 60%.
- If `cost.max_usd` is below `cost.spot_threshold_usd` (default 250), Spot is used.
- If the estimate exceeds `cost.max_usd`, submission fails fast.

## Batch queues and job definitions

Queues (required by the CLI):

- `pism-cpu-spot`
- `pism-gpu-spot`
- `pism-cpu-ondemand`

Job definitions:

- `pism-cpu`
- `pism-gpu`

Override these via environment variables:

- `PISM_CPU_SPOT_QUEUE`, `PISM_GPU_SPOT_QUEUE`, `PISM_CPU_ONDEMAND_QUEUE`
- `PISM_CPU_JOB_DEFINITION`, `PISM_GPU_JOB_DEFINITION`
- `PISM_RUNS_TABLE` for DynamoDB
- `PISM_LOG_GROUP` for the dashboard log links

## Notes

- GPU on-demand is not configured; GPU runs are Spot-only for now.
- Ensure the container images include `aws` CLI for S3 sync.
- For private GHCR images, copy to ECR or configure registry credentials.
- One job is forced per instance by requesting full instance vCPU/memory.
