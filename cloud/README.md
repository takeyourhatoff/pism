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
- Python 3.9+.
- Docker (for building images).

### 2) Create S3 buckets

```
aws s3 mb s3://pism-inputs
aws s3 mb s3://pism-results
```

Upload your inputs to:

```
s3://pism-inputs/<run-id>/
```

### 3) Install the CLI

```
python -m venv .venv
source .venv/bin/activate
pip install -e cloud
```

### 4) Build images and deploy infrastructure

```
pism-cloud deploy \
  --subnet-ids subnet-abc,subnet-def \
  --security-group-ids sg-123 \
  --input-bucket-arn arn:aws:s3:::pism-inputs \
  --output-bucket-arn arn:aws:s3:::pism-results
```

This builds and pushes CPU/GPU images to ECR, then deploys the CloudFormation stack.
The stack creates Batch queues, job definitions, a log group, and the DynamoDB table.
ECR repositories (`pism-cpu`, `pism-gpu`) are created automatically if missing.

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
pism-cloud dashboard --port 8080
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

## Container images

Default Dockerfiles live in `cloud/images/`:

- `cloud/images/Dockerfile.cpu`
- `cloud/images/Dockerfile.gpu`

Build and push both to ECR with:

```
pism-cloud build-images
```

These Dockerfiles build PISM from source; expect longer build times and customize as needed.
For other GPU types, set `--cuda-arch` when building images.

## Cost policy

- Spot is the default; set `compute.use_spot: false` to force CPU on-demand.
- Estimated costs are pulled from AWS Pricing and spot history at submit time.
- Set `cost.estimated_usd_per_hour` to override pricing if needed (requires `compute.instance`).
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
- One job is forced per instance by requesting full instance vCPU/memory.
- Cost estimation calls the AWS Pricing API (`pricing:GetProducts`).
