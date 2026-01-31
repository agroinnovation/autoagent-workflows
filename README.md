# AutoAgent Workflows

Deterministic workflow scripts for the AutoAgent platform.

## Structure

```
scrapers/     # Web scraping scripts (Playwright, requests)
pipelines/    # Data processing and transformation
jobs/         # Scheduled/cron jobs
```

## Relationship to autoagent-mcp

| Repo | Contains |
|------|----------|
| **autoagent-mcp** | Platform services, agent coordination, MCP tools |
| **autoagent-workflows** | Standalone scripts that agents develop and run |

Agents working in `autoagent-mcp` can read/write to this repo. Laptop Claude instances SFTP scraping scripts here for server agents to execute.

## Development

Scripts should be self-contained and runnable:

```bash
# Scrapers
python scrapers/example_scraper.py

# Pipelines
python pipelines/example_pipeline.py

# Jobs (typically called by cron)
./jobs/example_job.sh
```

## Guidelines

- Keep scripts stateless where possible
- Use environment variables for configuration
- Log to stdout/stderr (captured by platform)
- Exit codes: 0 = success, non-zero = failure
