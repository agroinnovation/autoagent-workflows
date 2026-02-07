# AutoAgent Workflows

Deterministic workflow scripts for the AutoAgent platform.

## Structure

```
scrapers/     # Web scraping scripts (Playwright, requests)
pipelines/    # Data processing and transformation
jobs/         # Scheduled/cron jobs
stock/        # Stock market data import and screening
analysis/     # Data analysis scripts (CSV-based, pandas)
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

## Stock Scripts

- `stock/stock_data_importer.py` - Bulk import stock fundamentals and technicals via API
- `stock/high_quality_value_screener.py` - Screen stocks by value/quality metrics (ROE, PEG, debt ratios)

## Analysis Scripts

- `analysis/climate_analysis.py` - Climate data analysis and visualization
- `analysis/heatpump_analysis.py` - Heat pump performance analysis
- `analysis/power_analysis.py` - Power consumption analysis

> **Note:** Analysis scripts expect CSV input files in the current working directory.

## Guidelines

- Keep scripts stateless where possible
- Use environment variables for configuration
- Log to stdout/stderr (captured by platform)
- Exit codes: 0 = success, non-zero = failure
