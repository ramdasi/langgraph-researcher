from typing import TypedDict, Annotated, Literal, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from googleapiclient.discovery import build
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
import requests
from bs4 import BeautifulSoup
import logging
import time
from datetime import datetime

# Rich imports for beautiful UI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.logging import RichHandler
from rich.tree import Tree
from rich import box
from rich.columns import Columns

# Initialize Rich console
console = Console()

# Set up rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    user_query: str
    llm_query: str
    urls: List[str]
    whiteboard: str
    search_results: List[Dict[str, Any]]
    final_report: str
    progress_data: Dict[str, Any]

# Configuration
GOOGLE_API_KEY = "AIzaSyBuQlFw-Uq0i7htWNFZUletKVSx3Bgz0Qs"
GOOGLE_CSE_ID = "868ab21c0787f4617"
MODEL = "qwen2.5:0.5b"
TEMPERATURE = 0.4

# Global progress tracking
progress_tracker = {
    "current_step": "",
    "total_urls": 0,
    "processed_urls": 0,
    "current_url": "",
    "start_time": None,
    "step_details": []
}

def update_progress(step: str, details: str = "", current_url: str = ""):
    """Update global progress tracker."""
    progress_tracker["current_step"] = step
    progress_tracker["current_url"] = current_url
    if details:
        progress_tracker["step_details"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {details}")

def create_progress_panel() -> Panel:
    """Create a progress panel showing current status."""
    content = []
    
    # Current step
    content.append(f"🔄 **Current Step:** {progress_tracker['current_step']}")
    
    # URL progress
    if progress_tracker['total_urls'] > 0:
        progress_bar = "█" * progress_tracker['processed_urls'] + "░" * (progress_tracker['total_urls'] - progress_tracker['processed_urls'])
        content.append(f"📊 **URL Progress:** {progress_tracker['processed_urls']}/{progress_tracker['total_urls']} [{progress_bar}]")
    
    # Current URL
    if progress_tracker['current_url']:
        content.append(f"🌐 **Processing:** {progress_tracker['current_url'][:60]}...")
    
    # Time elapsed
    if progress_tracker['start_time']:
        elapsed = time.time() - progress_tracker['start_time']
        content.append(f"⏱️  **Elapsed:** {elapsed:.1f}s")
    
    return Panel(
        "\n".join(content),
        title="[bold blue]Research Progress[/bold blue]",
        border_style="blue",
        padding=(1, 2)
    )

def open_url(url: str, max_chars: int = 1000) -> Dict[str, str]:
    """Fetch URL with progress tracking."""
    update_progress("Fetching Content", f"Opening {url}", url)
    
    try:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AgenticResearcher/2.0"}
        
        # The console.status block has been removed from here
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
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
            "text_excerpt": text,
            "original_length": original_length
        }
        
        progress_tracker['processed_urls'] += 1
        update_progress("Content Fetched", f"✅ Successfully fetched {len(text)} chars from {url}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        progress_tracker['processed_urls'] += 1
        update_progress("Fetch Error", f"❌ Error fetching {url}: {str(e)}")
        return {"title": url, "url": url, "text_excerpt": f"[ERROR] {e}", "original_length": 0}
def query_refinement(state: GraphState) -> GraphState:
    """Refine the user query for better search results."""
    update_progress("Query Refinement", "Optimizing search query...")
    
    with console.status("[bold yellow]🤖 LLM is refining your query..."):
        llm = ChatOllama(model=MODEL, temperature=0.4, repeat_penalty=1.1)
        
        system = SystemMessage(content=f"""
            You are a search query optimization expert. Given a user query, create an optimized search query that will return the most relevant results.
        

        
        User Query: {state['user_query']}
        
        Return only the optimized search query, nothing else.
        """)
        
        response = llm.invoke([system])
        refined_query = response.content.strip()
    
    # Show query refinement results
    table = Table(title="Query Refinement Results", box=box.ROUNDED)
    table.add_column("Type", style="cyan", width=15)
    table.add_column("Query", style="white")
    table.add_row("Original", state['user_query'])
    table.add_row("Refined", refined_query, style="green")
    console.print(table)
    
    update_progress("Query Refined", f"✅ Query optimized: '{refined_query}'")
    
    return {
        **state,
        "llm_query": refined_query,
        "whiteboard": f"Query refined from '{state['user_query']}' to '{refined_query}'\n"
    }

def search_google(state: GraphState) -> GraphState:
    """Search Google and fetch content from URLs with progress tracking."""
    update_progress("Google Search", "Searching for relevant content...")
    
    try:
        with console.status("[bold blue]🔍 Searching Google..."):
            service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
            query = state.get("llm_query", state["user_query"])
            
            res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=10).execute()
        
        items = []
        urls = []
        
        if "items" in res:
            progress_tracker['total_urls'] = len(res["items"])
            progress_tracker['processed_urls'] = 0
            
            console.print(f"\n[bold green]🎯 Found {len(res['items'])} search results![/bold green]")
            
            # Create a table showing search results
            results_table = Table(title="Search Results", box=box.ROUNDED)
            results_table.add_column("#", justify="center", width=3)
            results_table.add_column("Title", style="cyan", width=40)
            results_table.add_column("URL", style="blue", width=40)
            
            for i, r in enumerate(res["items"], 1):
                url = r.get("link", "")
                title = r.get("title", "")[:37] + "..." if len(r.get("title", "")) > 40 else r.get("title", "")
                short_url = url[:37] + "..." if len(url) > 40 else url
                results_table.add_row(str(i), title, short_url)
            
            console.print(results_table)
            console.print()
            
            # Process each URL with progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                
                task = progress.add_task("Processing URLs...", total=len(res["items"]))
                
                for i, r in enumerate(res["items"]):
                    url = r.get("link", "")
                    urls.append(url)
                    
                    progress.update(task, description=f"Processing URL {i+1}: {url[:30]}...")
                    
                    # Show current URL being processed
                    console.print(f"[bold yellow]📄 Processing:[/bold yellow] {r.get('title', 'Unknown Title')}")
                    console.print(f"[dim]🔗 {url}[/dim]")
                    
                    # Fetch content from URL
                    content = open_url(url, max_chars=10000)
                    
                    item = {
                        "title": r.get("title", ""),
                        "url": url,
                        "snippet": r.get("snippet", ""),
                        "content": content
                    }
                    items.append(item)
                    
                    # Show processing result
                    if content.get("text_excerpt", "").startswith("[ERROR]"):
                        console.print("[red]❌ Failed to fetch content[/red]")
                    else:
                        chars = len(content.get("text_excerpt", ""))
                        console.print(f"[green]✅ Fetched {chars} characters[/green]")
                    
                    progress.advance(task)
                    console.print()
        else:
            console.print("[red]❌ No search results found[/red]")
        
        update_progress("Search Complete", f"✅ Processed {len(items)} URLs")
        
        updated_whiteboard = state.get("whiteboard", "") + f"Found {len(items)} search results\n"
        
        return {
            **state,
            "search_results": items,
            "urls": urls,
            "whiteboard": updated_whiteboard
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        console.print(f"[red]❌ Search error: {e}[/red]")
        return {
            **state,
            "search_results": [],
            "urls": [],
            "whiteboard": state.get("whiteboard", "") + f"Search error: {e}\n"
        }

def generate_report(state: GraphState) -> GraphState:
    """Generate final research report with progress tracking."""
    update_progress("Report Generation", "Analyzing content and generating report...")
    
    with console.status("[bold magenta]🤖 LLM is analyzing content and generating report..."):
        llm = ChatOllama(model=MODEL, temperature=TEMPERATURE, repeat_penalty=1.2)
        
        # Prepare search results for the prompt
        search_summary = ""
        for i, item in enumerate(state.get("search_results", []), 1):
            search_summary += f"""
            Source {i}:
            Title: {item['title']}
            URL: {item['url']}
            Snippet: {item['snippet']}
            Content: {item['content']['text_excerpt']}
            
            """
        
        system = SystemMessage(content=f"""
            You are a research analyst. Create a short research report or one line answer for the user query.
            
            User Query: {state['user_query']}
            
            Internet references:
            {search_summary}
            
            Instructions:
            1. Write a well-structured report that directly addresses the user's query
            2. Include relevant information from the sources
            3. Use markdown formatting for better readability
            4. Highlight exact answer to user's query.
            
            Write a short research report of 200 words or answer of user query in markdown format:
        """)
        
        response = llm.invoke([system])
        report = response.content
    
    update_progress("Report Generated", "✅ Research report completed")
    
    updated_whiteboard = state.get("whiteboard", "") + "Generated final report\n"
    
    return {
        **state,
        "final_report": report,
        "whiteboard": updated_whiteboard
    }

def should_continue(state: GraphState) -> Literal["generate_report", "end"]:
    """Decide whether to continue or end."""
    if state.get("search_results"):
        return "generate_report"
    else:
        return "end"

# Create the workflow
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("query_refinement", query_refinement)
workflow.add_node("search_google", search_google)
workflow.add_node("generate_report", generate_report)

# Add edges
workflow.set_entry_point("query_refinement")
workflow.add_edge("query_refinement", "search_google")
workflow.add_conditional_edges(
    "search_google",
    should_continue,
    {
        "generate_report": "generate_report",
        "end": END
    }
)
workflow.add_edge("generate_report", END)

# Compile the graph
app = workflow.compile()

def run_research_agent(user_query: str) -> Dict[str, Any]:
    """Run the research agent with a user query."""
    # Reset progress tracker
    progress_tracker.update({
        "current_step": "",
        "total_urls": 0,
        "processed_urls": 0,
        "current_url": "",
        "start_time": time.time(),
        "step_details": []
    })
    
    initial_state = {
        "user_query": user_query,
        "llm_query": "",
        "urls": [],
        "whiteboard": f"Starting research for: {user_query}\n",
        "search_results": [],
        "final_report": "",
        "progress_data": {}
    }
    
    console.print(Panel(
        f"[bold white]🔍 Starting Research Agent[/bold white]\n\n"
        f"[cyan]Query:[/cyan] {user_query}\n"
        f"[dim]Model:[/dim] {MODEL}\n"
        f"[dim]Started:[/dim] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        title="[bold blue]Research Initialized[/bold blue]",
        border_style="blue"
    ))
    
    try:
        # Run the workflow
        final_state = app.invoke(initial_state)
        
        return {
            "success": True,
            "query": user_query,
            "refined_query": final_state.get("llm_query", ""),
            "urls": final_state.get("urls", []),
            "report": final_state.get("final_report", ""),
            "process_log": final_state.get("whiteboard", ""),
            "num_sources": len(final_state.get("search_results", [])),
            "elapsed_time": time.time() - progress_tracker["start_time"]
        }
        
    except Exception as e:
        logger.error(f"Research agent error: {e}")
        console.print(f"[red]❌ Research failed: {e}[/red]")
        return {
            "success": False,
            "error": str(e),
            "query": user_query
        }

def display_results(result: Dict[str, Any]):
    """Display research results with rich formatting."""
    if result["success"]:
        # Success summary
        console.print(Panel(
            f"[bold green]✅ Research Completed Successfully![/bold green]\n\n"
            f"[cyan]Original Query:[/cyan] {result['query']}\n"
            f"[cyan]Refined Query:[/cyan] {result['refined_query']}\n"
            f"[cyan]Sources Found:[/cyan] {result['num_sources']}\n"
            f"[cyan]Time Elapsed:[/cyan] {result['elapsed_time']:.2f} seconds",
            title="[bold green]Research Summary[/bold green]",
            border_style="green"
        ))
        
        # URLs found
        if result['urls']:
            console.print("\n[bold blue]🔗 Sources:[/bold blue]")
            for i, url in enumerate(result['urls'], 1):
                console.print(f"  {i}. {url}")
        
        # Main report
        console.print("\n" + "="*80)
        console.print(Panel.fit("[bold white]📋 RESEARCH REPORT[/bold white]", style="bold magenta"))
        console.print("="*80)
        
        # Display report as markdown
        markdown_report = Markdown(result['report'])
        console.print(markdown_report)
        
        console.print("="*80)
        
    else:
        console.print(Panel(
            f"[bold red]❌ Research Failed[/bold red]\n\n"
            f"[red]Error:[/red] {result['error']}\n"
            f"[cyan]Query:[/cyan] {result['query']}",
            title="[bold red]Error Report[/bold red]",
            border_style="red"
        ))

def interactive_chat_mode():
    """Run the agent in interactive chat mode."""
    # Welcome screen
    console.print(Panel(
        "[bold white]🤖 Welcome to AI Research Agent![/bold white]\n\n"
        "I'm your intelligent research assistant. Ask me anything and I'll:\n"
        "• Search the web for relevant information\n"
        "• Analyze multiple sources\n"
        "• Generate comprehensive reports\n\n"
        "[dim]Type 'quit', 'exit', or 'bye' to end the session[/dim]",
        title="[bold blue]AI Research Agent[/bold blue]",
        border_style="blue",
        padding=(1, 2)
    ))
    
    session_count = 0
    
    while True:
        try:
            console.print()
            # Get user input
            user_query = Prompt.ask(
                "[bold cyan]🔍 What would you like to research?[/bold cyan]",
                default=""
            ).strip()
            
            # Check for exit commands
            if user_query.lower() in ['quit', 'exit', 'bye', 'q']:
                console.print(Panel(
                    "[bold white]👋 Thank you for using AI Research Agent![/bold white]\n\n"
                    f"[cyan]Sessions completed:[/cyan] {session_count}\n"
                    "[dim]Have a great day![/dim]",
                    title="[bold blue]Goodbye![/bold blue]",
                    border_style="blue"
                ))
                break
            
            if not user_query:
                console.print("[yellow]⚠️  Please enter a research query.[/yellow]")
                continue
            
            session_count += 1
            console.print(f"\n[bold white]📊 Research Session #{session_count}[/bold white]")
            
            # Run the research
            result = run_research_agent(user_query)
            
            # Display results
            console.print()
            display_results(result)
            
            # Ask if user wants to continue
            console.print()
            
                
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
            if Confirm.ask("[bold red]🚪 Do you want to exit?[/bold red]", default=False):
                break
        except Exception as e:
            console.print(f"[red]❌ Unexpected error: {e}[/red]")
            logger.error(f"Chat mode error: {e}")

# Example usage and main execution
if __name__ == "__main__":
    try:
        # Start interactive mode
        interactive_chat_mode()
    except Exception as e:
        console.print(f"[red]❌ Fatal error: {e}[/red]")
        logger.error(f"Fatal error: {e}")