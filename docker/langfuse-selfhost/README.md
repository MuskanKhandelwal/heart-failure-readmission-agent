# Self-Hosted Langfuse (HIPAA-Aware Observability)

> For portfolio reference only — verify against current Langfuse docs before production use.

This directory documents a self-hosted [Langfuse](https://langfuse.com) v3 deployment
for tracing the LLM/agent components of the heart failure readmission planner.

## Why self-host for a healthcare deployment?

LLM tracing tools capture the full prompt, completion, retrieved context, and tool
I/O for every request. In this project those payloads can contain **PHI** —
patient demographics, diagnoses (ICD-9 428.x), medications, discharge details, and
free-text clinical context fed into the agent. That has direct compliance
consequences:

- **PHI in prompts cannot go to a SaaS vendor without a BAA.** Under HIPAA, any
  third party that receives, stores, or processes PHI on your behalf is a Business
  Associate and must sign a Business Associate Agreement. Sending traces to a
  managed/cloud observability endpoint that has not executed a BAA with your
  organization is an impermissible disclosure — regardless of the vendor's general
  security posture.
- **Self-hosting keeps trace data inside your VPC.** Running Langfuse on
  infrastructure you control (your cloud account / private network) means prompts,
  completions, and retrieved guideline context never leave the organization's
  trust boundary. There is no external data processor to contract with for the
  trace path, which removes one BAA dependency and shrinks the audit surface.
- **Data residency and retention stay under your control.** You own the Postgres /
  ClickHouse / object storage that backs Langfuse, so retention windows, access
  controls, encryption-at-rest, and deletion of PHI are governed by your existing
  policies rather than a vendor's defaults.

For a portfolio/demo running purely on **synthetic SynPUF data**, Langfuse Cloud is
fine and is what the app defaults to. This self-host stack documents the path a
real HIPAA deployment would take, and is referenced from the project README.

## What's here

- `docker-compose.yml` — a Langfuse v3 stack: `postgres`, `clickhouse`, `redis`,
  `minio` (S3-compatible object store), `langfuse-web`, and `langfuse-worker`.

## Quick start (local evaluation only)

```bash
cd docker/langfuse-selfhost

# 1. Replace every CHANGE_ME / dev secret below with strong, unique values.
#    Do NOT reuse the placeholder secrets in any shared or production environment.

# 2. Bring the stack up.
docker compose up -d

# 3. Open the UI and create a project to get API keys.
open http://localhost:3000
```

Point the app at the self-hosted instance via environment variables:

```bash
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

## Production hardening checklist (not exhaustive)

- Generate strong unique values for `NEXTAUTH_SECRET`, `SALT`,
  `ENCRYPTION_KEY` (must be 64 hex chars / 256-bit), and all database, Redis, and
  object-store credentials. Manage them with a secrets manager, not this file.
- Terminate TLS in front of `langfuse-web`; never expose plaintext HTTP carrying
  PHI.
- Run inside a private subnet/VPC with no public ingress to the data services
  (`postgres`, `clickhouse`, `redis`, `minio`).
- Enable encryption at rest and automated backups for Postgres, ClickHouse, and the
  object store; define a PHI retention/deletion policy.
- Restrict UI access (SSO/network policy) and enable audit logging.
- Pin image tags to specific digests and track Langfuse release notes — the
  compose schema and required env vars change between versions.
