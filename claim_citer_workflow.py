"""
Single Claim Citer - LangGraph workflow for finding evidence for a single claim.
by Garrett Jones
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Union, Tuple
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from firecrawl import Firecrawl
from firecrawl.v2.utils.error_handler import WebsiteNotSupportedError, FirecrawlError
from urllib.parse import urlparse
import json
import os
import hashlib
import time
from datetime import datetime
from operator import add

URL_DISPLAY_LENGTH = 80  # Maximum length for displaying URLs in print statements

def get_display_url(url: str) -> str:
    if len(url) > URL_DISPLAY_LENGTH:
        return f"{url[:URL_DISPLAY_LENGTH]}..."
    return url


def parse_json_array(content: str) -> List[Any]:
    json_start = content.find('[')
    json_end = content.rfind(']') + 1
    if json_start >= 0 and json_end > json_start:
        json_str = content[json_start:json_end]
        return json.loads(json_str)
    else:
        return json.loads(content)


def parse_json_dict(content: str) -> Dict[str, Any]:
    json_start = content.find('{')
    json_end = content.rfind('}') + 1
    if json_start >= 0 and json_end > json_start:
        json_str = content[json_start:json_end]
        return json.loads(json_str)
    else:
        return json.loads(content)


def keep_first_value(current: str, update: str) -> str:
    return current if current else update


class WorkerResult(TypedDict):
    url: str
    supports_claim: bool
    score: Optional[float]  # Score if supports_claim is True
    extracted_urls: Annotated[List[str], "URLs extracted from page if it doesn't directly support"]


class WorkerState(TypedDict):
    claim: str
    url: str
    content: Optional[str]
    worker_results: List[WorkerResult]


class SingleClaimState(TypedDict):
    claim: Annotated[str, keep_first_value]
    prior_search_queries: Annotated[List[str], "List of search queries used in prior searches"]
    new_search_queries: Annotated[List[str], "List of newly generated search queries not yet used"]
    candidate_urls: Annotated[List[str], "URLs to check"]
    checked_urls: Annotated[List[str], "URLs that have already been checked"]
    worker_results: Annotated[List[WorkerResult], add]
    supporting_urls: Annotated[List[Dict[str, Any]], "URLs that support the claim with scores"]
    final_url: Optional[str]
    alternative_urls: Annotated[List[str], "URLs with the same score as final_url"]
    iteration_count: int
    deferred_urls: Annotated[List[str], "URLs deferred from processing due to domain limits"]


class SingleClaimCiter:
    CONTENT_SIZE_LIMIT = 8000  # Maximum number of characters to use from scraped content
    DEFAULT_MAX_ITERATIONS = 2  # Default maximum number of search iterations
    MAX_URLS_PER_DOMAIN = 3  # Maximum number of URLs to process per domain per iteration
    
    def _generate_log_filename(self, prefix: str) -> str:
        timestamp = datetime.now().isoformat()
        hash_input = f"{timestamp}{os.getpid()}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:6]
        return f"{prefix}-{hash_value}.txt"
    
    def _ensure_logs_directory(self) -> None:
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
    
    def __init__(self, firecrawl: Firecrawl, model_name: str = "gpt-4o", temperature: float = 0.3):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.firecrawl = firecrawl
        
        self._ensure_logs_directory()
        # Generate a single hash for both log files to keep them linked
        timestamp = datetime.now().isoformat()
        hash_input = f"{timestamp}{os.getpid()}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:6]
        self.scrape_log_filename = f"scrape-{hash_value}.txt"
        self.model_output_log_filename = f"model-output-{hash_value}.txt"
        self.scrape_log_filepath = os.path.join("logs", self.scrape_log_filename)
        self.model_output_log_filepath = os.path.join("logs", self.model_output_log_filename)
        
        self.generate_search_template = ChatPromptTemplate.from_messages([
            ("system", "You are a research assistant expert at generating effective search queries."),
            ("human", """Given a claim and prior search queries, generate new search queries to find evidence.

Claim: {claim}

Prior search queries:
{prior_search_queries}

Generate 1-3 new search queries that might help find evidence for this claim. 
Consider different angles, keywords, and phrasings that haven't been tried yet.

Return your response as a JSON array of strings, each string being a search query.
Example: ["search query 1", "search query 2", "search query 3"]

If this is the first attempt (no prior search queries), generate initial search queries.""")
        ])
        
        self.evaluate_url_template = ChatPromptTemplate.from_messages([
            ("system", "You are a fact-checking expert. Be strict but fair in your evaluation."),
            ("human", """You are evaluating whether a web page supports a claim.

Claim: {claim}

URL: {url}

++++++++++++++++++++ Start Content (first {content_size_limit} characters) ++++++++++++++++++++
{content}
+++++++++++++++++++++ End Content (first {content_size_limit} characters) ++++++++++++++++++++

Determine:
1. Does this content directly support, verify, or provide evidence for the claim?
   - If YES: Provide a score from 0.0 to 1.0 based on how clearly/directly the claim is supported.
     - 1.0 = Direct, clear, strong evidence
     - 0.7-0.9 = Good evidence, clear support
     - 0.5-0.6 = Moderate evidence, some support
     - 0.0-0.4 = Weak or indirect evidence
  - If NO: score should be 0.0
   
2. Does the page contain URLs to other pages that would be good candidates for supporting the claim?
   - If YES: Extract and list those URLs

Return your response as JSON with this structure:
{{
  "supports_claim": true/false,
  "score": 0.0-1.0,
  "extracted_urls": [
    "url1",
    "url2",
    ...
  ]
}}
""")
        ])
        
        self.worker_graph = self._build_worker_graph()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(SingleClaimState)
        
        # Add nodes
        workflow.add_node("generate_search_queries", self._generate_search_queries_node)
        workflow.add_node("run_search_queries", self._run_search_queries_node)
        workflow.add_node("url_worker", self.worker_graph)  # Register worker graph as a node
        workflow.add_node("merge_url_results", self._merge_url_results_node)
        workflow.add_node("select_final_url", self._select_final_url_node)
        
        # Define edges
        workflow.add_edge(START, "generate_search_queries")
        workflow.add_edge("generate_search_queries", "run_search_queries")
        
        workflow.add_conditional_edges(
            "run_search_queries",
            self._start_url_workers,
            {
                "no_urls": "merge_url_results"
            }
        )
        
        workflow.add_edge("url_worker", "merge_url_results")
        
        workflow.add_conditional_edges(
            "merge_url_results",
            self._has_supporting_urls,
            {
                "has_support": "select_final_url",
                "max_iterations": "select_final_url",
                "no_support": "generate_search_queries",
            }
        )
        
        workflow.add_edge("select_final_url", END)
        
        return workflow.compile()
    
    def _build_worker_graph(self) -> StateGraph:
        workflow = StateGraph(WorkerState)
        
        # Add nodes
        workflow.add_node("scrape_url", self._scrape_url_node)
        workflow.add_node("evaluate_url", self._evaluate_url_node)
        
        # Define edges
        workflow.add_edge(START, "scrape_url")
        workflow.add_edge("scrape_url", "evaluate_url")
        workflow.add_edge("evaluate_url", END)
        
        return workflow.compile()
    
    def _generate_search_queries_node(self, state: SingleClaimState) -> SingleClaimState:
        claim = state["claim"]
        prior_search_queries = state["prior_search_queries"]
        
        print(f"\n🔍 Generating search queries for claim: '{claim}'")
        if prior_search_queries:
            print(f"   Prior search queries: {len(prior_search_queries)}")
        
        prior_search_queries_str = "\n".join([f"- {query}" for query in prior_search_queries]) if prior_search_queries else "None (first search)"
        
        messages = self.generate_search_template.format_messages(
            claim=claim,
            prior_search_queries=prior_search_queries_str
        )
        response = self.llm.invoke(messages)
        content = response.content
        
        with open(self.model_output_log_filepath, "a", encoding="utf-8") as log_file:
            log_file.write(f"===== Generate Search Queries =====\n")
            log_file.write(f"Claim: {claim}\n")
            log_file.write(f"Prior search queries: {prior_search_queries_str}\n")
            log_file.write(f"--- LLM Response ---\n")
            log_file.write(content)
            log_file.write("\n\n")
        
        search_queries = parse_json_array(content)
        
        if not isinstance(search_queries, list):
            search_queries = [str(search_queries)]
        
        search_queries = [str(q) for q in search_queries if q]
        
        state["new_search_queries"] = search_queries
        
        print(f"   Generated {len(search_queries)} search queries")
        for query in search_queries:
            print(f"   - {query}")
            
        return state
    
    def _run_search_queries_node(self, state: SingleClaimState) -> SingleClaimState:
        new_search_queries = state["new_search_queries"]
        if not new_search_queries:
            print("   ⚠️  No search queries available")
            return state
        
        print(f"\n🌐 Searching for URLs using Firecrawl...")
        candidate_urls = state["candidate_urls"]
        
        for query in new_search_queries:
            print(f"   Searching: '{query}'")
            search_results = self.firecrawl.search(query=query, limit=5)
            
            for result in search_results.web:
                url = result.url
                if url not in candidate_urls and url not in state["checked_urls"]:
                    candidate_urls.append(url)
        
        state["prior_search_queries"].extend(new_search_queries)
        state["new_search_queries"] = []
        
        state["candidate_urls"] = candidate_urls
        print(f"   Found {len(candidate_urls)} candidate URLs (total)")
        
        return state
    
    def _limit_urls_by_domain(self, urls: List[str]) -> Tuple[List[str], List[str]]:
        urls_by_domain: Dict[str, List[str]] = {}
        for url in urls:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc or parsed.path.split('/')[0] if parsed.path else url
                if not domain:
                    domain = url
                if domain not in urls_by_domain:
                    urls_by_domain[domain] = []
                urls_by_domain[domain].append(url)
            except Exception:
                # If parsing fails, use the URL itself as the domain
                domain = url
                if domain not in urls_by_domain:
                    urls_by_domain[domain] = []
                urls_by_domain[domain].append(url)
        
        urls_to_process = []
        deferred_urls = []
        
        for domain, domain_urls in urls_by_domain.items():
            urls_to_process.extend(domain_urls[:self.MAX_URLS_PER_DOMAIN])
            deferred_urls.extend(domain_urls[self.MAX_URLS_PER_DOMAIN:])
        
        return urls_to_process, deferred_urls
    
    def _start_url_workers(self, state: SingleClaimState) -> Union[List[Send], str]:
        urls_for_workers = state["candidate_urls"]
        if not urls_for_workers:
            return "no_urls"
        
        urls_to_process, deferred_urls = self._limit_urls_by_domain(urls_for_workers)
        state["deferred_urls"] = deferred_urls
        
        if not urls_to_process:
            return "no_urls"
        
        print(f"\n👷 Starting {len(urls_to_process)} workers for URLs (deferred {len(deferred_urls)} URLs)...")
        if deferred_urls:
            print(f"   Deferred URLs will be processed in next iteration")
        
        sends = []
        for url in urls_to_process:
            worker_state = {
                "claim": state["claim"],
                "url": url,
                "content": None,
                "worker_results": []
            }
            sends.append(Send("url_worker", worker_state))
        
        return sends
    
    def _scrape_url_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        url = state["url"]
        
        print(f"   📄 Scraping: {get_display_url(url)}")
        start_time = time.time()
        
        try:
            scrape_result = self.firecrawl.scrape(url, formats=["markdown"])
            state["content"] = scrape_result.markdown
            
            elapsed_time = time.time() - start_time
            print(f"   ✓ Scraped in {elapsed_time:.2f}s: {get_display_url(url)}")
            
            with open(self.scrape_log_filepath, "a", encoding="utf-8") as log_file:
                log_file.write(f"===== {url} =====\n")
                log_file.write(state["content"])
                log_file.write("\n\n")
        except WebsiteNotSupportedError as e:
            elapsed_time = time.time() - start_time
            print(f"   ✗ Website not supported: {get_display_url(url)}")
            print(f"   Error: {e}")
            print(f"   Scraping took {elapsed_time:.2f}s")
            state["content"] = ""
            
            with open(self.scrape_log_filepath, "a", encoding="utf-8") as log_file:
                log_file.write(f"===== {url} =====\n")
                log_file.write(f"Error: Website not supported - {e}\n\n")
        except FirecrawlError as e:
            # Handle API errors (timeouts, rate limits, etc.)
            elapsed_time = time.time() - start_time
            error_msg = str(e)
            status_code = getattr(e, 'status_code', None)
            if status_code:
                print(f"   ✗ Firecrawl API error ({status_code}): {get_display_url(url)}")
            else:
                print(f"   ✗ Firecrawl API error: {get_display_url(url)}")
            print(f"   Error: {error_msg}")
            print(f"   Scraping took {elapsed_time:.2f}s")
            state["content"] = ""
            
            with open(self.scrape_log_filepath, "a", encoding="utf-8") as log_file:
                log_file.write(f"===== {url} =====\n")
                log_file.write(f"Error: Firecrawl API error (status: {status_code}) - {error_msg}\n\n")
        except Exception as e:
            # Catch any other unexpected errors
            elapsed_time = time.time() - start_time
            print(f"   ✗ Unexpected error scraping {get_display_url(url)}")
            print(f"   Error: {e}")
            print(f"   Scraping took {elapsed_time:.2f}s")
            state["content"] = ""
            
            with open(self.scrape_log_filepath, "a", encoding="utf-8") as log_file:
                log_file.write(f"===== {url} =====\n")
                log_file.write(f"Error: Unexpected error - {type(e).__name__}: {e}\n\n")
        return state
    
    def _evaluate_url_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        url = state["url"]
        content = state["content"]
        
        if not content:
            print(f"   ✗ No content available for evaluation")
            result: WorkerResult = {
                "url": url,
                "supports_claim": False,
                "score": None,
                "extracted_urls": []
            }
            state["worker_results"].append(result)
            return state
        
        print(f"   🔍 Evaluating: {get_display_url(url)}")
        
        messages = self.evaluate_url_template.format_messages(
            claim=state["claim"],
            url=url,
            content=content[:self.CONTENT_SIZE_LIMIT],
            content_size_limit=self.CONTENT_SIZE_LIMIT
        )
        response = self.llm.invoke(messages)
        content_response = response.content
        
        with open(self.model_output_log_filepath, "a", encoding="utf-8") as log_file:
            log_file.write(f"===== Evaluate URL =====\n")
            log_file.write(f"Claim: {state['claim']}\n")
            log_file.write(f"URL: {url}\n")
            log_file.write(f"--- LLM Response ---\n")
            log_file.write(content_response)
            log_file.write("\n\n")
        
        evaluation = parse_json_dict(content_response)
        
        supports_claim = evaluation.get("supports_claim", False)
        score = evaluation.get("score")
        extracted_urls = evaluation.get("extracted_urls", [])
        
        if supports_claim:
            if score:
                score = float(score)
                if score < 0.0 or score > 1.0:
                    score = max(0.0, min(1.0, score))
            else:
                raise ValueError(f"Score is missing for URL: {url}")
        else:
            score = 0.0
        
        result: WorkerResult = {
            "url": url,
            "supports_claim": supports_claim,
            "score": score,
            "extracted_urls": extracted_urls
        }
        
        if supports_claim:
            print(f"   ✓ Supports claim (score: {score:.2f}): {get_display_url(url)}")
        else:
            print(f"   ✗ Does not support claim: {get_display_url(url)}")
        
        if extracted_urls:
            print(f"   → Extracted {len(extracted_urls)} URLs from page: {get_display_url(url)}")
        else:
            print(f"   → No URLs extracted from page: {get_display_url(url)}")

        state["worker_results"] = [result]
        return state
    
    def _merge_url_results_node(self, state: SingleClaimState) -> SingleClaimState:
        worker_results = state["worker_results"]
        supporting_urls = state["supporting_urls"]

        processed_urls = [result["url"] for result in worker_results]
        state["checked_urls"].extend(processed_urls)
        state["candidate_urls"] = []
        deferred_urls = state["deferred_urls"]
        if deferred_urls:
            candidate_urls.extend(deferred_urls)
            state["deferred_urls"] = []
            print(f"Added {len(deferred_urls)} deferred URLs back to candidates")

        checked_urls = state["checked_urls"]
        candidate_urls = state["candidate_urls"]
        
        print(f"\n📊 Gathering results from {len(worker_results)} workers...")
        
        extracted_url_count = 0
        for worker_result in worker_results:
            url = worker_result["url"]
            
            supports = worker_result["supports_claim"]
            score = worker_result["score"]
            extracted = worker_result["extracted_urls"]
            
            if supports and score is not None:
                supporting_urls.append({
                    "url": url,
                    "score": score
                })
                print(f"   ✓ (score: {score:.2f}) {get_display_url(url)}")
            
            for extracted_url in extracted:
                extracted_url_count += 1
                if extracted_url not in candidate_urls and extracted_url not in checked_urls:
                    candidate_urls.append(extracted_url)
        
        print(f"   New supporting URLs: {len(supporting_urls)}")
        print(f"   Total extracted URLs: {extracted_url_count}")
        print(f"   Net extracted URLs: {len(candidate_urls)}")
        
        state["iteration_count"] += 1
        print(f"\n🔄 Iteration {state['iteration_count']}/{self.DEFAULT_MAX_ITERATIONS}")
        
        state["worker_results"] = []

        return state
    
    def _select_final_url_node(self, state: SingleClaimState) -> SingleClaimState:
        supporting_urls = state["supporting_urls"]
        
        if supporting_urls:
            best_url = max(supporting_urls, key=lambda x: x["score"])
            best_score = best_url["score"]
            state["final_url"] = best_url["url"]
            
            alternative_urls = [
                url_dict["url"]
                for url_dict in supporting_urls 
                if url_dict["score"] == best_score and url_dict["url"] != state["final_url"]
            ]
            state["alternative_urls"] = alternative_urls
            
            print(f"\n✅ Selected final URL: {state['final_url']}")
            print(f"   Score: {best_score:.2f}")
            if alternative_urls:
                print(f"   Alternative URLs with same score ({best_score:.2f}): {len(alternative_urls)}")
                for alt_url in alternative_urls:
                    print(f"   - {get_display_url(alt_url)}")
        else:
            state["final_url"] = None
            state["alternative_urls"] = []
            print(f"\n❌ No supporting URL found")
        
        return state
    
    def _has_urls_to_evaluate(self, state: SingleClaimState) -> str:
        if state["candidate_urls"]:
            return "has_urls"
        else:
            return "no_urls"
    
    def _has_supporting_urls(self, state: SingleClaimState) -> str:
        if state["supporting_urls"]:
            return "has_support"
        elif state["iteration_count"] >= self.DEFAULT_MAX_ITERATIONS:
            return "max_iterations"
        else:
            return "no_support"
    
    def find_citation(self, claim: str) -> Optional[str]:
        initial_state: SingleClaimState = {
            "claim": claim,
            "prior_search_queries": [],
            "new_search_queries": [],
            "candidate_urls": [],
            "checked_urls": [],
            "worker_results": [],
            "supporting_urls": [],
            "final_url": None,
            "alternative_urls": [],
            "iteration_count": 0,
            "deferred_urls": []
        }
        
        print(f"\n🚀 Starting citation search for claim: '{claim}'")
        print("=" * 80)
        
        final_state = self.graph.invoke(initial_state)
        return {
            "final_url": final_state["final_url"],
            "alternative_urls": final_state["alternative_urls"]
        }


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python claim_citer_workflow.py <claim_text>")
        print('Example: python claim_citer_workflow.py "Men are taller than women on average"')
        sys.exit(1)
    
    claim = sys.argv[1]
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set it as an environment variable: export OPENAI_API_KEY=your_key")
        sys.exit(1)
    
    if not os.getenv("FIRECRAWL_API_KEY"):
        print("Error: FIRECRAWL_API_KEY not found in environment variables")
        print("Please set it as an environment variable: export FIRECRAWL_API_KEY=your_key")
        print("You can get an API key at https://www.firecrawl.dev/")
        sys.exit(1)
    
    firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))
    
    claim_citer = SingleClaimCiter(firecrawl=firecrawl)
    result = claim_citer.find_citation(claim)
    
    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    
    final_url = result.get("final_url", None)
    if final_url:
        print(f"✅ Found supporting URL: {final_url}")
        alternative_urls = result.get("alternative_urls", [])
        if alternative_urls:
            print("Alternative URLs:")
            for alt_url in alternative_urls:
                print(f"   - {alt_url}")
    else:
        print("❌ No supporting URL found")
    print("=" * 80)


if __name__ == "__main__":
    main()

