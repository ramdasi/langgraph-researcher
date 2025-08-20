# Agentic Researcher v1.0

🔍 **AI-Powered Research Assistant with Real-Time Process Visibility**

An intelligent research automation tool that combines LangGraph agents with live operation tracking, providing detailed insights into every step of the research process.
![alt text](https://github.com/ramdasi/langgraph-researcher/blob/main/v1/sample-research.png)
![alt text](https://github.com/ramdasi/langgraph-researcher/blob/main/v1/sample-result.png)
## ✨ Features

- **Autonomous Web Research** - Google Custom Search integration with intelligent result filtering
- **Content Extraction** - Advanced web scraping with HTML parsing and text processing
- **Knowledge Management** - Automatic note-taking with relevance scoring and organization
- **Real-Time Monitoring** - Live visualization of agent operations, tool calls, and processing steps
- **Interactive CLI** - Beautiful terminal interface with live updates and progress tracking

## 📋 Requirements

### Dependencies
```bash
pip install langchain-ollama langgraph rich beautifulsoup4 requests google-api-python-client aiohttp
```

### Services
1. **Ollama** - Install from [ollama.ai](https://ollama.ai), then: `ollama pull qwen2.5:7b`
2. **Google Custom Search API** - Get API key and Custom Search Engine ID

### Configuration
1. Create `prompts.py` with your system prompt:
```python
main_prompt = "Your AI research assistant system prompt here..."
```

2. Edit API credentials in `researcher.py`:
```python
GOOGLE_API_KEY = "your-google-api-key"
GOOGLE_CSE_ID = "your-cse-id"
```

## 🚀 Quick Start

```bash
# Start Ollama
ollama serve

# Run the researcher
python researcher.py
```

## 💻 Usage

### Research Examples
```
🔍 Research Query ➤ "latest quantum computing breakthroughs 2024"
🔍 Research Query ➤ "compare renewable energy policies across EU countries"
```

### Commands
- `/clear` - Reset research data and logs
- `/stats` - Show performance statistics  
- `/quit` - Exit session

## 🖥️ Interface

The live interface displays:
- **Header** - Session stats and elapsed time
- **Current Operations** - Active tool calls and processing steps
- **Performance Stats** - Success rates and operation counts
- **Research Tasks** - Task status and progress

## 🛠️ Research Tools

- `web_search` - Google search with result filtering
- `open_url` - Web content extraction and analysis
- `add_research_task` - Task creation and management
- `save_note` - Smart note-taking with relevance scoring
- `get_notes` - Retrieve saved research notes
- `mark_task_complete` - Task workflow management

## ⚙️ Configuration

```python
MODEL = "qwen2.5:7b"        # Ollama model
TEMPERATURE = 0.4           # Response creativity
max_results = 3            # Search results per query
max_chars = 1000           # Max chars per webpage
```

## 📊 Monitoring

Real-time tracking of:
- Operation success/failure rates
- Average execution times
- Research task completion
- Note relevance scores
- Live agent decision-making

Access detailed analytics with `/stats` command for operation breakdowns, performance metrics, and session summaries.