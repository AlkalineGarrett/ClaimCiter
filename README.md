# Claim Citer

A LangGraph-based system for automatically finding evidence and citations for claims by searching the web, scraping content, and evaluating relevance using LLMs.

by Garrett Jones

## Overview

Claim Citer provides two implementations for finding supporting URLs for claims:

1. **`claim_citer_workflow.py`** - A structured workflow with explicit search, scrape, and evaluation phases
2. **`claim_citer_agent.py`** - An agent-based approach where an LLM autonomously decides when to search and scrape using tools

Both implementations combine:
- **LLM-powered search query generation** - Generates search queries based on the claim
- **Web search** - Uses Firecrawl to search the web for relevant URLs
- **Content scraping** - Extracts and processes web page content
- **Intelligent evaluation** - Uses LLMs to score how well each URL supports the claim
- **Iterative refinement** - Continues searching and evaluating until a good citation is found

## Installation

```bash
pip install -r requirements.txt
```

## Setup

You'll need API keys for two services:

1. **OpenAI API Key** - For LLM-powered search query generation and URL evaluation
   - Get your key from https://platform.openai.com/api-keys
   - Set as environment variable: `export OPENAI_API_KEY=your_key`

2. **Firecrawl API Key** - For web search and content scraping
   - Get your key from https://www.firecrawl.dev/
   - Set as environment variable: `export FIRECRAWL_API_KEY=your_key`

## Usage

### Workflow Approach

Run the workflow script with a claim as an argument:

```bash
python claim_citer_workflow.py "Men are taller than women on average"
```

**Output:**
- **Final URL**: The best supporting URL found (highest score)
- **Alternative URLs**: Other URLs with the same score as the final URL
- **Logs**: Detailed logs saved in the `logs/` directory:
  - `scrape-{hash}.txt` - All scraped content
  - `model-output-{hash}.txt` - All LLM responses

### Agent Approach

Run the agent script with a claim as an argument:

```bash
python claim_citer_agent.py "Men are taller than women on average"
```

**Output:**
- **Status**: "found" or "not_found"
- **Best URL**: The best supporting URL found
- **Alternative URLs**: Other URLs that support the claim
- **Logs**: Detailed logs saved in the `logs/` directory:
  - `model-output-{hash}.txt` - All LLM responses and tool calls
  - `tool-output-{hash}.txt` - All tool execution results
- **Timing**: Individual scrape times and total process time

## Configuration

### Workflow Configuration

Key parameters in `ClaimCiterWorkflow` class:

- `CONTENT_SIZE_LIMIT = 8000` - Maximum characters from scraped content to evaluate
- `DEFAULT_MAX_ITERATIONS = 2` - Maximum number of search iterations
- `MAX_URLS_PER_DOMAIN = 3` - Maximum URLs to process per domain per iteration
- `model_name = "gpt-4o"` - LLM model to use (default: gpt-4o)
- `temperature = 0.3` - LLM temperature for deterministic outputs

### Agent Configuration

Key parameters in `ClaimCiterAgent` class:

- `MAX_TURNS = 20` - Maximum number of agent turns
- `CONTENT_SIZE_LIMIT = 8000` - Maximum characters from scraped content to evaluate
- `model_name = "gpt-4o"` - LLM model to use (default: gpt-4o)
- `temperature = 0.3` - LLM temperature for deterministic outputs
