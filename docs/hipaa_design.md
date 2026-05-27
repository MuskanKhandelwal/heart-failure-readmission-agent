# HIPAA-Aware Design Notes

This file will capture the project-level decisions and controls for handling synthetic patient data, self-hosted tracing, and production deployment considerations.

## Key design goals

- No real PHI is stored or transmitted in this repository.
- Langfuse tracing configuration is separated from user data ingestion.
- Self-hosted Docker Compose deployment is documented for HIPAA-aware production.
