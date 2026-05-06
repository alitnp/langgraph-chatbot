# LangGraph Chatbot

A Python-based AI chatbot built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain). This project demonstrates how to build an assistant with memory, tool calling capabilities, and state management.

## Features

- **LangGraph State Management:** Leverages state graphs for intelligent conversational routing.
- **Tool Integration (Tavily):** Uses the Tavily search tool to browse the internet and answer questions.
- **Ollama Integration:** Uses local LLMs via Ollama for privacy and speed.
- **Session Memory:** Includes built-in memory savers (and persistent SQLite-based memory) to remember context within and across chat sessions.

## Project Structure

- `main.py` - Entry point for the project.
- `chatbot/app.py` - Core chatbot application featuring LangGraph state definition, tool routing, and CLI interaction.
- `chatbot/chatbot_memory.py` / `chatbot_persist_memory.py` - Implementations of chatbot memory.
- `common/llms.py` - LLM configuration and initialization.
- `tools/search.py` - Custom tool configurations.

## Prerequisites

- Python >= 3.12
- Local installation of [Ollama](https://ollama.com/) (with your preferred model pulled)
- [Tavily API Key](https://tavily.com/) for search capabilities

## Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   cd langraphtest
   ```

2. Optionally, create and activate a virtual environment.

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This project can also be managed by `uv` or `poetry` via the `pyproject.toml`.*

4. Set up your environment variables:
   Create a `.env` file in the root directory and add your API keys:
   ```
   TAVILY_API_KEY=your_tavily_api_key
   ```

## Usage

You can run the interactive chatbot CLI directly:

```bash
python -m chatbot.app
```

Interact with the bot in your terminal. Type `exit` or `quit` to end the conversation. 
