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
import json
import os
import sys


CONTENT_SIZE_LIMIT = 8000


def _search_impl(firecrawl: Firecrawl, query: str) -> str:
    """Search for websites using Firecrawl."""
    search_results = firecrawl.search(query=query, limit=10)
    return f"{search_results}"


def _scrape_impl(firecrawl: Firecrawl, content_limit: int, urls: List[str]) -> Dict[str, Any]:
    """Scrape content from a list of URLs using Firecrawl."""
    results = []
    for url in urls:
        scrape_result = firecrawl.scrape(url, formats=["markdown"])
        content = scrape_result.markdown[:content_limit]
        results.append({"url": url, "content": content})
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
        
        self.search_tool = self._create_search_tool()
        self.scrape_tool = self._create_scrape_tool()
        self.tools = [self.search_tool, self.scrape_tool]
        
        self.llm_with_tools = self.llm.bind_tools(self.tools)
    
    def _create_search_tool(self):
        @tool
        def search(query: str) -> str:
            """Search for websites using Firecrawl."""
            return _search_impl(self.firecrawl, query)
        return search
    
    def _create_scrape_tool(self):
        @tool
        def scrape(urls: List[str]) -> Dict[str, Any]:
            """Scrape content from a list of URLs using Firecrawl."""
            return _scrape_impl(self.firecrawl, CONTENT_SIZE_LIMIT, urls)
        return scrape
    
    def _run_agent_step(self, state: AgentState) -> AgentState:
        """Agent node that decides what action to take and calls tools."""
        claim = state["claim"]
        messages = state["messages"]
        turn_count = state["turn_count"]
        
        nudge = ""
        if turn_count >= self.MAX_TURNS - 3:
            nudge = f"\n\n⚠️ IMPORTANT: You are on turn {turn_count} of {self.MAX_TURNS}. Wrap up soon."
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are ClaimCiterAgent, an AI agent that finds URLs supporting claims."),
            ("human", """Find a URL that supports this claim: {claim}

You have access to these tools:
1. search(query: str) - Search for websites
2. scrape(urls: List[str]) - Scrape content from URLs (up to {content_limit} chars per URL)

Guidance:
* Write out reasoning between each step
* Continue calling tools until you find a supporting URL or reach the limit
* Be selective about which URLs to scrape
* Track which URLs you've already checked to avoid duplicates

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
        
        response = self.llm_with_tools.invoke(new_messages)
        state["messages"] = new_messages + [response]
        state["turn_count"] = turn_count + 1
        
        return state
    
    def _run_tools(self, state: AgentState) -> AgentState:
        """Execute tools and add results to messages."""
        messages = state["messages"]
        last_message = messages[-1]
        
        tool_map = {tool.name: tool for tool in self.tools}
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool = tool_map[tool_name]
            result = tool.invoke(tool_args)
            
            tool_message = ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            messages.append(tool_message)
        
        state["messages"] = messages
        return state
    
    def find_citation(self, claim: str) -> Dict[str, Any]:
        state: AgentState = {
            "claim": claim,
            "messages": [],
            "turn_count": 0,
            "result": None
        }
        
        while True:
            state = self._run_agent_step(state)
            
            if state["turn_count"] >= self.MAX_TURNS:
                break
            
            last_message = state["messages"][-1]
            if not last_message.tool_calls:
                break
            
            state = self._run_tools(state)
        
        result = self._extract_result(state["messages"])
        return result
    
    def _extract_result(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        result = json.loads(messages[-1].content)
        return {
            "status": result["status"],
            "bestUrl": result["bestUrl"],
            "alternativeUrls": result["alternativeUrls"]
        }

def main():
    claim = sys.argv[1]
    
    firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))
    agent = ClaimCiterAgent(firecrawl=firecrawl)

    # The result is already printed in find_citation
    result = agent.find_citation(claim)


if __name__ == "__main__":
    main()
