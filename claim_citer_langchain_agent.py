"""
ClaimCiterAgent - An AI agent for finding evidence URLs that support claims.
by Garrett Jones
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from firecrawl import Firecrawl
from firecrawl.v2.utils.error_handler import WebsiteNotSupportedError, FirecrawlError
import json
import os
import sys
import hashlib
import time
from datetime import datetime


CONTENT_SIZE_LIMIT = 8000


def _search_impl(firecrawl: Firecrawl, query: str) -> str:
    """Search for websites using Firecrawl."""
    try:
        search_results = firecrawl.search(query=query, limit=10)
        return f"{search_results}"
    except Exception as e:
        return f"Exception in search tool: {e}"


def _scrape_impl(firecrawl: Firecrawl, content_limit: int, urls: List[str]) -> Dict[str, Any]:
    """Scrape content from a list of URLs using Firecrawl."""
    results = []
    for url in urls:
        start_time = time.time()
        try:
            scrape_result = firecrawl.scrape(url, formats=["markdown"])
            content = scrape_result.markdown[:content_limit]
            elapsed = time.time() - start_time
            print(f"   ⏱️  Scraped {url[:60]}... in {elapsed:.2f}s")
            results.append({"url": url, "content": content, "error": None})
        except WebsiteNotSupportedError as e:
            elapsed = time.time() - start_time
            print(f"   ⏱️  Scrape failed (not supported) {url[:60]}... in {elapsed:.2f}s")
            results.append({"url": url, "content": "", "error": f"Website not supported: {e}"})
        except FirecrawlError as e:
            elapsed = time.time() - start_time
            print(f"   ⏱️  Scrape failed (Firecrawl error) {url[:60]}... in {elapsed:.2f}s")
            results.append({"url": url, "content": "", "error": f"Firecrawl error: {e}"})
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"   ⏱️  Scrape failed (unexpected error) {url[:60]}... in {elapsed:.2f}s")
            results.append({"url": url, "content": "", "error": f"Unexpected error: {e}"})
    return {"results": results}


class AgentState(TypedDict, total=False):
    claim: str
    messages: Annotated[List[BaseMessage], "Conversation history with tool results"]
    turn_count: int
    result: Optional[Dict[str, Any]]


class ClaimCiterAgent:
    MAX_TURNS = 20
    
    def __init__(self, firecrawl: Firecrawl, model_name: str = "gpt-4o", temperature: float = 0.3):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.firecrawl = firecrawl
        
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().isoformat()
        hash_input = f"{timestamp}{os.getpid()}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:6]
        self.model_output_log_filename = f"model-output-{hash_value}.txt"
        self.tool_output_log_filename = f"tool-output-{hash_value}.txt"
        self.model_output_log_filepath = os.path.join("logs", self.model_output_log_filename)
        self.tool_output_log_filepath = os.path.join("logs", self.tool_output_log_filename)
        
        self.search_tool = self._create_search_tool()
        self.scrape_tool = self._create_scrape_tool()
        self.tools = [self.search_tool, self.scrape_tool]
        
        self.llm_with_tools = self.llm.bind_tools(self.tools)
    
    def _create_search_tool(self):
        @tool
        def search(query: str) -> str:
            """
            Search for websites using Firecrawl.
            
            Args:
                query: A single search query string
                
            Returns:
                String representation of search results with all available information (URLs, titles, descriptions, snippets, etc.)
            """
            return _search_impl(self.firecrawl, query)
        
        return search
    
    def _create_scrape_tool(self):
        @tool
        def scrape(urls: List[str]) -> Dict[str, Any]:
            """
            Scrape content from a list of URLs using Firecrawl.
            Returns up to 8000 characters per URL.
            
            Args:
                urls: List of URLs to scrape
                
            Returns:
                Dictionary with 'results' key containing list of dicts with 'url' and 'content' keys
            """
            return _scrape_impl(self.firecrawl, CONTENT_SIZE_LIMIT, urls)
        
        return scrape
    
    def _run_agent_step(self, state: AgentState) -> AgentState:
        """Agent node that decides what action to take and calls tools."""
        claim = state["claim"]
        messages = state["messages"]
        turn_count = state["turn_count"]
        
        nudge = ""
        if turn_count >= self.MAX_TURNS - 3:
            nudge = f"\n\n⚠️ IMPORTANT: You are on turn {turn_count} of {self.MAX_TURNS}. You need to wrap up soon. Provide your final answer with the best supporting URL you've found, or conclude that no supporting URL was found."
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are ClaimCiterAgent, an AI agent that finds URLs supporting claims. Always explain your reasoning."),
            ("human", """Find a URL that supports this claim: {claim}

You have access to these tools:
1. search(query: str) - Search for websites. Returns search results with URLs, titles, descriptions, and snippets.
2. scrape(urls: List[str]) - Scrape content from URLs (up to {content_limit} chars are provided for each).

In addition to using the tools above, the consider the following helpful actions

1. generate search queries to send to the search tool
2. decide which urls from a search result list are worth scraping (given the goal)
3. decide which urls from a scraped page are worth scraping
4. conclude that the goal of finding a url that supports a claim is satisfied
5. conclude that the goal is unlikely to be satisfied and terminate.

Other actions can be taken also as appropriate.

Guidance:
* Write out reasoning between each step, including an analysis of the prior step’s results, and what the best next action is.
* Continue calling tools and performing helpful actions (in the best order) until the limits described in the rules are reached.
* Be selective about which URLs to scrape.
* Track which URLs you've already checked to avoid duplicates.
* Don’t terminate early because it is generally known that a claim is untrue. Do your best to find a source that supports the claim in order to be helpful to the user.
  * In a case where it seems that a claim is generally considered untrue, it may be helpful to seek out urls that are more likely to represent contrarian viewpoints.

Provide the final answer in this JSON format:
{{"status": "found" or "not_found", "bestUrl": "url or null", "alternativeUrls": ["url1", "url2"]}}
{nudge}""")
        ])
        
        if not messages:
            new_messages = prompt_template.format_messages(
                claim=claim,
                content_limit=CONTENT_SIZE_LIMIT,
                nudge=nudge
            )
        else:
            new_messages = messages
            if nudge:
                new_messages.append(HumanMessage(content=nudge))
        
        print(f"\n🤖 LLM Turn {turn_count + 1}: Invoking LLM...")
        response = self.llm_with_tools.invoke(new_messages)
        
        if response.content:
            content = response.content
            if len(content) > 1000:
                print(f"💭 LLM Response: {content[:500]}...{content[-500:]}")
            else:
                print(f"💭 LLM Response: {content}")
        
        if response.tool_calls:
            print(f"🔧 LLM Tool Calls ({len(response.tool_calls)}):")
            for tool_call in response.tool_calls:
                print(f"   - {tool_call['name']}({tool_call['args']})")
        else:
            print("✅ LLM provided final answer (no tool calls)")
        
        with open(self.model_output_log_filepath, "a", encoding="utf-8") as log_file:
            log_file.write(f"===== LLM Turn {turn_count + 1} =====\n")
            log_file.write(f"Claim: {claim}\n")
            if response.content:
                log_file.write(f"--- LLM Response ---\n")
                log_file.write(response.content)
                log_file.write("\n")
            if response.tool_calls:
                log_file.write(f"--- Tool Calls ({len(response.tool_calls)}) ---\n")
                for tool_call in response.tool_calls:
                    log_file.write(f"{tool_call['name']}({tool_call['args']})\n")
            log_file.write("\n\n")
        
        state["messages"] = new_messages + [response]
        state["turn_count"] = turn_count + 1
        
        return state
    
    def _run_tools(self, state: AgentState) -> AgentState:
        """Execute tools and add results to messages."""
        messages = state["messages"]
        last_message = messages[-1]
        
        print(f"\n⚙️  Executing {len(last_message.tool_calls)} tool(s)...")
        tool_map = {tool.name: tool for tool in self.tools}
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"   🔨 Executing: {tool_name}({tool_args})")
            tool = tool_map[tool_name]
            result = tool.invoke(tool_args)
            print(f"   ✓ {tool_name} completed")
            
            with open(self.tool_output_log_filepath, "a", encoding="utf-8") as log_file:
                log_file.write(f"===== {tool_name} =====\n")
                log_file.write(f"Arguments: {tool_args}\n")
                log_file.write(f"--- Result ---\n")
                log_file.write(str(result))
                log_file.write("\n\n")
            
            tool_message = ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            messages.append(tool_message)
        
        state["messages"] = messages
        return state
    
    def find_citation(self, claim: str) -> Dict[str, Any]:
        """Main entry point to find citation for a claim."""
        state: AgentState = {
            "claim": claim,
            "messages": [],
            "turn_count": 0,
            "result": None
        }
        
        print(f"\n🚀 ClaimCiterAgent starting for claim: '{claim}'")
        print("=" * 80)
        
        start_time = time.time()
        
        while True:
            state = self._run_agent_step(state)
            
            if state["turn_count"] >= self.MAX_TURNS:
                break
            
            last_message = state["messages"][-1]
            if not last_message.tool_calls:
                break
            
            state = self._run_tools(state)
        
        total_elapsed = time.time() - start_time
        
        result = self._extract_result(state["messages"])
        state["result"] = result
        
        print("\n📝 Conversation:")
        print("-" * 80)
        for i, msg in enumerate(state["messages"], 1):
            print(f"{i}. {msg.__class__.__name__}: {msg.content[:200]}...")
        
        print("\n" + "=" * 80)
        print("RESULT")
        print("=" * 80)
        print(json.dumps(result, indent=2))
        print("=" * 80)
        print(f"\n⏱️  Total process time: {total_elapsed:.2f}s")
        print("=" * 80)
        
        return result
    
    def _extract_result(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Extract final result from LLM's last message."""
        content = messages[-1].content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        json_str = content[json_start:json_end]
        result = json.loads(json_str)
        return {
            "status": result["status"],
            "bestUrl": result["bestUrl"],
            "alternativeUrls": result["alternativeUrls"]
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python claim_citer_langchain_agent.py <claim_text>")
        print('Example: python claim_citer_langchain_agent.py "Men are taller than women on average"')
        sys.exit(1)
    
    claim = sys.argv[1]
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    
    if not os.getenv("FIRECRAWL_API_KEY"):
        print("Error: FIRECRAWL_API_KEY not found in environment variables")
        sys.exit(1)
    
    firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))
    agent = ClaimCiterAgent(firecrawl=firecrawl)

    # The result is already printed in find_citation
    result = agent.find_citation(claim)


if __name__ == "__main__":
    main()

