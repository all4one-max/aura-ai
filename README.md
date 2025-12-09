# Aura AI

Aura AI is an agentic shopping assistant built with LangGraph, LangChain, and SQLModel.

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (fast Python package installer and resolver)
- OpenAI API Key
- SerpApi API Key (for Google Shopping search)

## Setup & Installation

1.  **Navigate to the project directory.**

2.  **Install Dependencies with `uv`**:
    This project uses `uv` for dependency management.
    ```bash
    uv sync
    ```
    This will create a virtual environment (`.venv`) and install all required packages specified in `pyproject.toml`.

3.  **Environment Variables**:
    Copy the template file to create your `.env` file:
    ```bash
    cp .env.template .env
    ```
    Then, open `.env` and add your API keys:
    ```env
    OPENAI_API_KEY=sk-proj-...
    SERPAPI_API_KEY=your_serpapi_key_here
    ```
    
    Get your SerpApi key from: https://serpapi.com/

## Database Setup

The project uses a lightweight SQLite database (`database.db`). The database tables are automatically created when you run the application or the verification script.

To explicitly initialize or check the database logic, you can check `app/database.py`, but typically running the verification script handles the necessary initialization via `create_db_and_tables()`.

## Running & Testing

To verify the agent logic and flow, use the provided verification script. This script runs through several test scenarios (recommendation, general QnA, etc.).

```bash
uv run verify_graph.py
```

This command will:
1.  Initialize the SQLite database.
2.  Run the orchestration graph.
3.  Simulate user interactions to verify the "Context Agent" and downstream agents.

## Project Structure

- `app/`: Core application code (agents, tools, schema, database).
- `verify_graph.py`: Main entry point for testing the agent workflow.
- `pyproject.toml`: Project dependencies managed by `uv`.