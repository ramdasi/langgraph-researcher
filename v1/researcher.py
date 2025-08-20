#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agentic Researcher v2.0 - Detailed Internal Processing Visibility
- Shows every tool call, parameter, and result in real-time
- Granular progress tracking for searches, URL fetching, content analysis
- Live streaming of agent thoughts and decision-making process
- Detailed breakdown of each research step with timing
"""

import asyncio
import aiohttp
import json
import time
import sys
import uuid
from datetime import datetime
from typing import TypedDict, Annotated, Literal, List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# Rich imports for beautiful CLI
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text
from rich.layout import Layout
from rich.align import Align

# LangChain and LangGraph imports
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Additional imports
from bs4 import BeautifulSoup
import requests
import textwrap
from googleapiclient.discovery import build
from contextlib import asynccontextmanager
from prompts import main_prompt
# Global console instance
console = Console()


MODEL = "qwen2.5:7b"
TEMPERATURE = 0.4
GOOGLE_API_KEY = ""
GOOGLE_CSE_ID = ""
# ---------------------------
# Data Models
# ---------------------------

class StepType(Enum):
    AGENT_THINKING = "agent_thinking"
    TOOL_CALL = "tool_call"
    TOOL_EXECUTION = "tool_execution"
    TOOL_RESULT = "tool_result"
    CONTENT_ANALYSIS = "content_analysis"
    SYNTHESIS = "synthesis"
    ERROR = "error"

@dataclass
class ProcessingStep:
    id: str
    step_type: StepType
    timestamp: datetime
    title: str
    details: str
    status: str = "running"  # running, completed, failed
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolCall:
    tool_name: str
    parameters: Dict[str, Any]
    call_id: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ResearchTask:
    id: str
    description: str
    status: str = "pending"
    steps: List[ProcessingStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    current_step: Optional[str] = None

# ---------------------------
# Global State with Detailed Tracking
# ---------------------------

class DetailedResearchState:
    def __init__(self):
        self.current_steps: List[ProcessingStep] = []
        self.completed_steps: List[ProcessingStep] = []
        self.active_tool_calls: Dict[str, ToolCall] = {}
        self.tasks: List[ResearchTask] = []
        self.notes: List[Dict] = []
        self.session_start = datetime.now()
        self.current_agent_action = None
        
    def add_step(self, step_type: StepType, title: str, details: str, **metadata) -> str:
        step_id = str(uuid.uuid4())[:8]
        step = ProcessingStep(
            id=step_id,
            step_type=step_type,
            timestamp=datetime.now(),
            title=title,
            details=details,
            metadata=metadata
        )
        self.current_steps.append(step)
        return step_id
    
    def update_step(self, step_id: str, status: str, duration: float = None, **metadata):
        for step in self.current_steps:
            if step.id == step_id:
                step.status = status
                step.duration = duration
                step.metadata.update(metadata)
                if status in ["completed", "failed"]:
                    self.current_steps.remove(step)
                    self.completed_steps.append(step)
                break
    
    def add_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        call_id = str(uuid.uuid4())[:8]
        tool_call = ToolCall(tool_name, parameters, call_id)
        self.active_tool_calls[call_id] = tool_call
        
        # Also add as a processing step
        params_str = json.dumps(parameters, indent=2)
        self.add_step(
            StepType.TOOL_CALL,
            f"Calling {tool_name}",
            f"Parameters:\n{params_str}",
            tool_name=tool_name,
            call_id=call_id
        )
        return call_id
    
    def complete_tool_call(self, call_id: str, result: Any, duration: float):
        if call_id in self.active_tool_calls:
            del self.active_tool_calls[call_id]
        
        # Find and update the corresponding step
        for step in self.current_steps:
            if step.metadata.get("call_id") == call_id:
                result_preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                step.details += f"\n\nResult:\n{result_preview}"
                self.update_step(step.id, "completed", duration)
                break

# Global state
state = DetailedResearchState()

# ---------------------------
# Enhanced Display with Step-by-Step Visibility
# ---------------------------

class DetailedDisplay:
    def __init__(self):
        self.layout = Layout()
        self.setup_layout()
    
    def setup_layout(self):
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=6)
        )
        
        self.layout["main"].split_row(
            Layout(name="current_ops", ratio=3),
            Layout(name="stats", ratio=1)
        )
    
    def get_header_panel(self):
        elapsed = datetime.now() - state.session_start
        active_count = len(state.current_steps)
        completed_count = len(state.completed_steps)
        
        header_text = Text()
        header_text.append("🔍 ", style="blue")
        header_text.append("Agentic Researcher v2.0", style="bold cyan")
        header_text.append(f" • Active: {active_count}", style="yellow")
        header_text.append(f" • Completed: {completed_count}", style="green")
        header_text.append(f" • Elapsed: {str(elapsed).split('.')[0]}", style="dim")
        
        return Panel(Align.center(header_text), style="cyan")
    
    def get_current_operations_panel(self):
        if not state.current_steps:
            return Panel("🔄 Waiting for operations...", title="[bold]Current Operations", style="blue")
        
        content = []
        
        # Show current agent action if any
        if state.current_agent_action:
            agent_text = Text()
            agent_text.append("🧠 ", style="magenta")
            agent_text.append("Agent: ", style="bold magenta")
            agent_text.append(state.current_agent_action, style="white")
            content.append(Panel(agent_text, style="magenta", padding=(0, 1)))
        
        # Show active steps
        for step in state.current_steps[-5:]:  # Show last 5 active steps
            elapsed = (datetime.now() - step.timestamp).total_seconds()
            
            # Create step display
            step_content = []
            
            # Title with timing
            title_text = Text()
            if step.step_type == StepType.TOOL_CALL:
                title_text.append("🔧 ", style="yellow")
            elif step.step_type == StepType.CONTENT_ANALYSIS:
                title_text.append("📊 ", style="cyan")
            elif step.step_type == StepType.AGENT_THINKING:
                title_text.append("🧠 ", style="magenta")
            else:
                title_text.append("⚡ ", style="blue")
            
            title_text.append(step.title, style="bold white")
            title_text.append(f" ({elapsed:.1f}s)", style="dim")
            step_content.append(title_text)
            
            # Details
            if step.details:
                details_lines = step.details.split('\n')[:3]  # Show first 3 lines
                for line in details_lines:
                    if line.strip():
                        detail_text = Text()
                        detail_text.append("  ", style="dim")
                        detail_text.append(line.strip()[:80], style="dim")
                        step_content.append(detail_text)
            
            # Status indicator
            status_text = Text()
            status_text.append("  Status: ", style="dim")
            if step.status == "running":
                status_text.append("⏳ Running", style="yellow")
            elif step.status == "completed":
                status_text.append("✅ Completed", style="green")
            else:
                status_text.append("❌ Failed", style="red")
            step_content.append(status_text)
            
            content.append(Group(*step_content))
            content.append(Text())  # Spacing
        
        return Panel(
            Group(*content) if content else "No active operations",
            title="[bold]Live Operations",
            style="blue",
            padding=(1, 1)
        )
    
    def get_stats_panel(self):
        # Recent completed operations
        recent_completed = state.completed_steps[-5:]
        
        stats_content = []
        
        # Summary stats
        stats_table = Table(show_header=False, show_edge=False, pad_edge=False, box=None)
        stats_table.add_column("Label", style="cyan", width=12)
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("Tasks", str(len(state.tasks)))
        stats_table.add_row("Notes", str(len(state.notes)))
        stats_table.add_row("Operations", str(len(state.completed_steps)))
        stats_table.add_row("Active", str(len(state.current_steps)))
        
        stats_content.append(Panel(stats_table, title="Stats", style="green", padding=(0, 1)))
        
        # Recent completions
        if recent_completed:
            completion_content = []
            for step in recent_completed[-3:]:
                comp_text = Text()
                comp_text.append("✅ ", style="green")
                comp_text.append(step.title[:25], style="white")
                if step.duration:
                    comp_text.append(f" ({step.duration:.1f}s)", style="dim")
                completion_content.append(comp_text)
            
            stats_content.append(Panel(
                Group(*completion_content),
                title="Recent",
                style="yellow",
                padding=(0, 1)
            ))
        
        return Group(*stats_content)
    
    def get_footer_panel(self):
        if not state.tasks:
            return Panel("📝 No research tasks yet", title="Tasks", style="dim")
        
        # Show current tasks
        task_content = []
        for task in state.tasks[-3:]:
            task_text = Text()
            task_text.append(f"📋 {task.id}: ", style="cyan")
            task_text.append(task.description[:50], style="white")
            task_text.append(f" [{task.status}]", style="dim")
            task_content.append(task_text)
        
        return Panel(
            Group(*task_content),
            title="[bold]Research Tasks",
            style="magenta"
        )
    
    def render(self):
        self.layout["header"].update(self.get_header_panel())
        self.layout["current_ops"].update(self.get_current_operations_panel())
        self.layout["stats"].update(self.get_stats_panel())
        self.layout["footer"].update(self.get_footer_panel())
        return self.layout

# Global display
display = DetailedDisplay()

# ---------------------------
# Enhanced Tools with Detailed Logging
# ---------------------------

@tool
def web_search(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Search with detailed step-by-step logging."""
    call_id = state.add_tool_call("web_search", {"query": query, "max_results": max_results})
    start_time = time.time()
    
    try:
        # Step 1: Initialize search
        step_id = state.add_step(
            StepType.TOOL_EXECUTION,
            f"Initializing Google Search",
            f"Query: '{query}'\nMax results: {max_results}"
        )
        

        
        if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
            state.update_step(step_id, "failed", error="Missing credentials")
            state.complete_tool_call(call_id, [], time.time() - start_time)
            return []
        
        state.update_step(step_id, "completed", time.time() - start_time)
        
        # Step 2: Execute API call
        api_step = state.add_step(
            StepType.TOOL_EXECUTION,
            "Executing Google Search API",
            f"Sending request to Google Custom Search...\nQuery: {query}"
        )
        
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=max_results).execute()
        
        state.update_step(api_step, "completed", time.time() - start_time)
        
        # Step 3: Process results
        process_step = state.add_step(
            StepType.CONTENT_ANALYSIS,
            "Processing search results",
            "Extracting titles, URLs, and snippets..."
        )
        
        items = []
        if "items" in res:
            for i, r in enumerate(res["items"]):
                items.append({
                    "title": r.get("title", ""),
                    "url": r.get("link", ""),
                    "snippet": r.get("snippet", "")
                })
                
                # Update progress
                if i % 2 == 0:  # Update every 2 items
                    state.update_step(
                        process_step,
                        "running",
                        details=f"Processing search results...\nProcessed {i+1}/{len(res['items'])} results"
                    )
        
        duration = time.time() - start_time
        state.update_step(
            process_step,
            "completed",
            duration,
            results_found=len(items)
        )
        
        state.complete_tool_call(call_id, f"Found {len(items)} results", duration)
        return items
        
    except Exception as e:
        duration = time.time() - start_time
        error_step = state.add_step(
            StepType.ERROR,
            f"Search failed: {str(e)}",
            f"Error during Google search for '{query}'"
        )
        state.update_step(error_step, "failed", duration)
        state.complete_tool_call(call_id, f"Error: {str(e)}", duration)
        return []

@tool
def open_url(url: str, max_chars: int = 1000) -> Dict[str, str]:
    """Fetch URL with detailed progress tracking."""
    max_chars = min(max_chars, 1000)

    short_url = textwrap.shorten(url, width=50, placeholder="...")
    call_id = state.add_tool_call("open_url", {"url": short_url, "max_chars": max_chars})
    start_time = time.time()
    
    try:
        # Step 1: Initialize request
        init_step = state.add_step(
            StepType.TOOL_EXECUTION,
            f"Connecting to {short_url}",
            f"Full URL: {url}\nMax characters: {max_chars:,}"
        )
        
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AgenticResearcher/2.0"}
        
        # Step 2: HTTP request
        request_step = state.add_step(
            StepType.TOOL_EXECUTION,
            f"Downloading content",
            f"Making HTTP request to {short_url}..."
        )
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        state.update_step(init_step, "completed", time.time() - start_time)
        state.update_step(
            request_step,
            "completed",
            time.time() - start_time,
            status_code=response.status_code,
            content_length=len(response.text)
        )
        
        # Step 3: Parse content
        parse_step = state.add_step(
            StepType.CONTENT_ANALYSIS,
            "Parsing HTML content",
            f"Processing {len(response.text):,} characters of HTML..."
        )
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove unwanted elements
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.extract()
        
        # Extract text
        text = " ".join(soup.get_text(" ").split())
        original_length = len(text)
        
        if len(text) > max_chars:
            text = text[:max_chars] + " … [truncated]"
        
        # Extract title
        title = (soup.title.string or "").strip() if soup.title else url
        
        result = {
            "title": title,
            "url": url,
            "text_excerpt": text
        }
        
        duration = time.time() - start_time
        state.update_step(
            parse_step,
            "completed",
            duration,
            original_length=original_length,
            final_length=len(text),
            title=title
        )
        
        state.complete_tool_call(call_id, f"Extracted {len(text):,} chars", duration)
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        error_step = state.add_step(
            StepType.ERROR,
            f"Failed to fetch {short_url}",
            f"Error: {str(e)}"
        )
        state.update_step(error_step, "failed", duration)
        state.complete_tool_call(call_id, f"Error: {str(e)}", duration)
        return {"title": url, "url": url, "text_excerpt": f"[ERROR] {e}"}

@tool
def add_research_task(task: str) -> List[str]:
    """Add research task with logging."""
    call_id = state.add_tool_call("add_research_task", {"task": task})
    start_time = time.time()
    
    step_id = state.add_step(
        StepType.TOOL_EXECUTION,
        f"Adding research task",
        f"Task: {task}"
    )
    
    # Create new task
    new_task = ResearchTask(
        id=str(uuid.uuid4())[:8],
        description=task.strip()
    )
    state.tasks.append(new_task)
    
    task_list = [f"{t.id}: {t.description}" for t in state.tasks]
    
    duration = time.time() - start_time
    state.update_step(step_id, "completed", duration, task_id=new_task.id)
    state.complete_tool_call(call_id, f"Task {new_task.id} created", duration)
    
    return task_list

@tool
def save_note(source_title: str, url: str, excerpt: str, relevance_score: float = 0.7) -> Dict[str, Any]:
    """Save note with detailed processing."""
    call_id = state.add_tool_call("save_note", {
        "source_title": source_title[:50] + "...",
        "url": url,
        "relevance_score": relevance_score
    })
    start_time = time.time()
    
    step_id = state.add_step(
        StepType.CONTENT_ANALYSIS,
        f"Saving research note",
        f"Title: {source_title}\nURL: {url}\nExcerpt length: {len(excerpt)} chars\nRelevance: {relevance_score}"
    )
    
    note = {
        "id": str(uuid.uuid4())[:8],
        "title": source_title.strip(),
        "url": url.strip(),
        "excerpt": textwrap.shorten(excerpt.strip(), width=800, placeholder=" …"),
        "relevance_score": relevance_score,
        "created_at": datetime.now().isoformat()
    }
    
    state.notes.append(note)
    
    duration = time.time() - start_time
    state.update_step(step_id, "completed", duration, note_id=note["id"])
    state.complete_tool_call(call_id, f"Note {note['id']} saved", duration)
    
    return note

@tool
def get_notes() -> List[Dict[str, Any]]:
    """Get all notes."""
    call_id = state.add_tool_call("get_notes", {})
    start_time = time.time()
    
    step_id = state.add_step(
        StepType.TOOL_EXECUTION,
        "Retrieving saved notes",
        f"Found {len(state.notes)} notes in memory"
    )
    
    duration = time.time() - start_time
    state.update_step(step_id, "completed", duration)
    state.complete_tool_call(call_id, f"Retrieved {len(state.notes)} notes", duration)
    
    return state.notes

@tool
def list_research_tasks() -> List[str]:
    """List current tasks."""
    call_id = state.add_tool_call("list_research_tasks", {})
    start_time = time.time()
    
    step_id = state.add_step(
        StepType.TOOL_EXECUTION,
        "Listing research tasks",
        f"Found {len(state.tasks)} tasks"
    )
    
    task_list = [f"{t.id}: {t.description} [{t.status}]" for t in state.tasks]
    
    duration = time.time() - start_time
    state.update_step(step_id, "completed", duration)
    state.complete_tool_call(call_id, f"Listed {len(state.tasks)} tasks", duration)
    
    return task_list

@tool
def clear_notes() -> int:
    """Clear all notes."""
    call_id = state.add_tool_call("clear_notes", {})
    start_time = time.time()
    
    count = len(state.notes)
    
    step_id = state.add_step(
        StepType.TOOL_EXECUTION,
        f"Clearing {count} notes",
        "Removing all saved research notes from memory"
    )
    
    state.notes.clear()
    
    duration = time.time() - start_time
    state.update_step(step_id, "completed", duration, cleared_count=count)
    state.complete_tool_call(call_id, f"Cleared {count} notes", duration)
    
    return 0

@tool
def mark_task_complete(task_id: str) -> str:
    """Mark a research task as complete by its ID."""
    call_id = state.add_tool_call("mark_task_complete", {"task_id": task_id})
    start_time = time.time()

    # Find the task by its ID
    task = next((t for t in state.tasks if t.id == task_id), None)

    step_id = state.add_step(
        StepType.TOOL_EXECUTION,
        "Marking a research task as complete",
        f"Attempting to mark task with ID {task_id} as complete"
    )

    if task:
        if task.status != "completed":
            task.status = "completed"
            message = f"Task {task_id}: '{task.description}' marked as complete."
        else:
            message = f"Task {task_id} is already complete."
    else:
        message = f"Error: Task with ID {task_id} not found."

    duration = time.time() - start_time
    state.update_step(step_id, "completed", duration, task_id=task_id, new_status="completed")
    state.complete_tool_call(call_id, message, duration)

    return message

# ---------------------------
# Enhanced LangGraph with Agent Visibility
# ---------------------------

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str

def agent_node(state_dict: GraphState) -> GraphState:
    """Agent with detailed thought process logging."""
    
    # Log agent thinking
    state.current_agent_action = f"Processing: {state_dict['user_input'][:50]}..."
    thinking_step = state.add_step(
        StepType.AGENT_THINKING,
        "Agent analyzing request",
        f"Input: {state_dict['user_input']}\nAnalyzing research requirements..."
    )
    
    start_time = time.time()
    
    try:
        llm = ChatOllama(model=MODEL, temperature=TEMPERATURE, repeat_penalty=1.5)
        tools = [web_search, open_url, add_research_task, save_note, get_notes, list_research_tasks, clear_notes, mark_task_complete]
        llm_with_tools = llm.bind_tools(tools)

        if not state_dict.get("messages"):
            system = SystemMessage(content=main_prompt)
            messages = [system, HumanMessage(content=state_dict["user_input"])]
        else:
            messages = state_dict["messages"]

        # Update thinking status
        state.current_agent_action = "Generating response with language model..."
        state.update_step(
            thinking_step,
            "running",
            details=f"Input: {state_dict['user_input']}\nSending to language model for processing..."
        )

        response = llm_with_tools.invoke(messages)
        
        # Log if agent is making tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            state.current_agent_action = f"Planning {len(response.tool_calls)} tool calls..."
            for i, tc in enumerate(response.tool_calls):
                call_details = f"Tool: {tc['name']}\nArgs: {json.dumps(tc['args'], indent=2)}"
                state.add_step(
                    StepType.TOOL_CALL,
                    f"Agent requesting {tc['name']}",
                    call_details,
                    tool_name=tc['name'],
                    call_index=i
                )
        else:
            state.current_agent_action = "Generating final response..."
        
        duration = time.time() - start_time
        state.update_step(thinking_step, "completed", duration)
        print(response.content)
        return {**state_dict, "messages": messages + [response]}
        
    except Exception as e:
        duration = time.time() - start_time
        state.current_agent_action = f"Error: {str(e)}"
        state.update_step(thinking_step, "failed", duration, error=str(e))
        raise

def should_continue(state_dict: GraphState) -> Literal["continue", "tools", "end"]:
    """Enhanced decision logic with logging."""
    last = state_dict["messages"][-1]
    
    if hasattr(last, "tool_calls") and last.tool_calls:
        state.current_agent_action = f"Executing {len(last.tool_calls)} tool calls..."
        return "tools"
    
    if isinstance(last, ToolMessage):
        state.current_agent_action = "Processing tool results..."
        return "continue"
    
    if isinstance(last, AIMessage):
        content = (last.content or "").lower()
        # if any(phrase in content for phrase in ["ready for q&a", "## ready for q&a", "KEY POINTS"]):
        #     state.current_agent_action = "Research completed ✅"
        return "end"
    
    return "continue"

def create_graph():
    """Create the research graph."""
    workflow = StateGraph(GraphState)
    tools = [web_search, open_url, add_research_task, save_note, get_notes, list_research_tasks, clear_notes, mark_task_complete]

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "agent", "tools": "tools", "end": END},
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()

# ---------------------------
# Enhanced CLI with Live Step Tracking
# ---------------------------

async def run_research_with_live_tracking(app, user_input: str):
    """Run research with detailed live tracking."""
    
    def research_task():
        state_dict = {"messages": [], "user_input": user_input}
        try:
            final_state = app.invoke(state_dict,{"recursion_limit": 100})
            return final_state
        except Exception as e:
            console.print(f"[red]Research error: {e}[/red]")
            return state_dict
    
    # Clear previous state for new research
    state.current_steps.clear()
    state.active_tool_calls.clear()
    state.current_agent_action = None
    
    # Run with live display
    with Live(display.render(), refresh_per_second=4, console=console) as live:
        import threading
        
        # Run research in background thread
        result = [None]
        exception = [None]
        
        def run_research():
            try:
                result[0] = research_task()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=run_research)
        thread.start()
        
        # Update display while research runs
        while thread.is_alive():
            live.update(display.render())
            time.sleep(0.25)
        
        thread.join()
        
        # Final update
        live.update(display.render())
        
        if exception[0]:
            raise exception[0]
        
        return result[0]

def display_final_report(messages: List[Any]):
    """Display the final research report."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not isinstance(msg, ToolMessage) and msg.content:
            # Final completion step
            state.add_step(
                StepType.SYNTHESIS,
                "Research report generated",
                f"Final report ready with {len(msg.content)} characters"
            )
            
            console.print("\n" + "═" * 100)
            console.print(Panel(
                Markdown(msg.content),
                title="[bold cyan]🔍 Research Report Complete[/bold cyan]",
                border_style="cyan",
                padding=(1, 2)
            ))
            console.print("═" * 100 + "\n")
            
            # Show final statistics
            total_time = (datetime.now() - state.session_start).total_seconds()
            stats_table = Table(title="Research Session Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            stats_table.add_column("Details", style="dim")
            
            stats_table.add_row("Total Time", f"{total_time:.1f}s", "Session duration")
            stats_table.add_row("Operations", str(len(state.completed_steps)), "Completed successfully")
            stats_table.add_row("Tasks Created", str(len(state.tasks)), "Research tasks")
            stats_table.add_row("Notes Saved", str(len(state.notes)), "Knowledge extracted")
            
            console.print(Panel(stats_table, title="[bold green]Session Summary[/bold green]", style="green"))
            break

async def main():
    """Enhanced main loop with comprehensive logging."""
    # Welcome banner
    welcome_text = Text()
    welcome_text.append("🔍 ", style="blue")
    welcome_text.append("Agentic Researcher v2.0\n", style="bold cyan")
    welcome_text.append("Detailed Internal Process Visibility", style="cyan")
    welcome_text.append("\n\nFeatures:", style="white")
    welcome_text.append("\n• Live tool call tracking", style="dim")
    welcome_text.append("\n• Step-by-step operation logging", style="dim")
    welcome_text.append("\n• Real-time content analysis", style="dim")
    welcome_text.append("\n• Comprehensive error reporting", style="dim")
    
    console.print(Panel(
        Align.center(welcome_text),
        style="cyan",
        padding=(1, 3)
    ))
    
    console.print("\n[dim]Commands: '/clear' - reset state, '/stats' - show statistics, '/quit' - exit[/dim]")
    
    app = create_graph()
    
    while True:
        try:
            console.print()
            user_input = console.input("[bold blue]🔍 Research Query[/bold blue] ❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]👋 Research session ended[/yellow]")
            break

        if not user_input:
            continue
            
        if user_input.lower() in {"/quit", "/exit", "/q"}:
            console.print("[yellow]👋 Research session ended[/yellow]")
            break
            
        if user_input.lower() == "/clear":
            # Clear all state
            state.notes.clear()
            state.tasks.clear()
            state.completed_steps.clear()
            state.current_steps.clear()
            state.active_tool_calls.clear()
            state.current_agent_action = None
            console.print("[green]✅ Cleared all research data and logs[/green]")
            continue
            
        if user_input.lower() in {"/stats", "/status"}:
            # Show detailed statistics
            stats_layout = Layout()
            stats_layout.split_row(
                Layout(name="operations"),
                Layout(name="performance")
            )
            
            # Operations breakdown
            op_table = Table(title="Operations Breakdown")
            op_table.add_column("Type", style="cyan")
            op_table.add_column("Count", style="white")
            op_table.add_column("Avg Duration", style="yellow")
            op_table.add_column("Success Rate", style="green")
            
            # Group operations by type
            op_stats = {}
            for step in state.completed_steps:
                step_type = step.step_type.value
                if step_type not in op_stats:
                    op_stats[step_type] = {"count": 0, "durations": [], "successes": 0}
                
                op_stats[step_type]["count"] += 1
                if step.duration:
                    op_stats[step_type]["durations"].append(step.duration)
                if step.status == "completed":
                    op_stats[step_type]["successes"] += 1
            
            for op_type, stats in op_stats.items():
                avg_duration = sum(stats["durations"]) / len(stats["durations"]) if stats["durations"] else 0
                success_rate = (stats["successes"] / stats["count"]) * 100 if stats["count"] > 0 else 0
                
                op_table.add_row(
                    op_type.replace("_", " ").title(),
                    str(stats["count"]),
                    f"{avg_duration:.2f}s" if avg_duration > 0 else "-",
                    f"{success_rate:.1f}%"
                )
            
            # Performance metrics
            perf_table = Table(title="Performance Metrics")
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Value", style="white")
            
            session_duration = (datetime.now() - state.session_start).total_seconds()
            perf_table.add_row("Session Duration", f"{session_duration:.1f}s")
            perf_table.add_row("Total Operations", str(len(state.completed_steps)))
            perf_table.add_row("Active Operations", str(len(state.current_steps)))
            perf_table.add_row("Research Tasks", str(len(state.tasks)))
            perf_table.add_row("Notes Collected", str(len(state.notes)))
            
            if state.notes:
                avg_relevance = sum(note.get("relevance_score", 0) for note in state.notes) / len(state.notes)
                perf_table.add_row("Avg Note Relevance", f"{avg_relevance:.2f}")
            
            stats_layout["operations"].update(Panel(op_table, style="blue"))
            stats_layout["performance"].update(Panel(perf_table, style="green"))
            
            console.print(stats_layout)
            continue

        # Execute research with live tracking
        try:
            state.current_agent_action = f"Starting research: {user_input[:50]}..."
            console.print(f"\n[cyan]🚀 Starting research session...[/cyan]")
            
            final_state = await run_research_with_live_tracking(app, user_input)
            
            # Display final report
            display_final_report(final_state.get("messages", []))
            
        except Exception as e:
            console.print(f"\n[red]❌ Research failed: {str(e)}[/red]")
            
            # Log the error
            state.add_step(
                StepType.ERROR,
                "Research session failed",
                f"Error: {str(e)}\nInput: {user_input}"
            )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Research session interrupted[/yellow]")
        sys.exit(0)
        