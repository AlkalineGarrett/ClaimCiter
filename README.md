# Claim Citer

A LangGraph-based workflow for automatically finding evidence and citations for claims by searching the web, scraping content, and evaluating relevance using LLMs.

by Garrett Jones

## Overview

Claim Citer uses an iterative search and evaluation process to find the best supporting URL for a given claim. It combines:
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

Run the script with a claim as an argument:

```bash
python claim_citer.py "Men are taller than women on average"
```

### Output

The script provides:
- **Final URL**: The best supporting URL found (highest score)
- **Alternative URLs**: Other URLs with the same score as the final URL
- **Logs**: Detailed logs saved in the `logs/` directory:
  - `scrape-{hash}.txt` - All scraped content
  - `model-output-{hash}.txt` - All LLM responses

## Configuration

Key parameters in `SingleClaimCiter` class:

- `CONTENT_SIZE_LIMIT = 8000` - Maximum characters from scraped content to evaluate
- `DEFAULT_MAX_ITERATIONS = 2` - Maximum number of search iterations
- `MAX_URLS_PER_DOMAIN = 3` - Maximum URLs to process per domain per iteration
- `model_name = "gpt-4o"` - LLM model to use (default: gpt-4o)
- `temperature = 0.3` - LLM temperature for deterministic outputs
