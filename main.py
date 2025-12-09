"""
FastAPI server for Aura AI chat application.
Provides REST API endpoints for LangGraph-based shopping assistant.
"""

from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import psycopg

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import asyncpg

from api_models import ChatRequest, ChatResponse
from app.graph import create_graph
from app.database import create_db_and_tables
from app.config import DATABASE_URL
from langchain_core.messages import HumanMessage
import uuid

# Global variable to store compiled graph
compiled_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles startup and shutdown events.
    """
    # Startup: Initialize database and compile graph
    print("🚀 Starting up Aura AI server...")

    # Compile LangGraph (expensive operation - do once at startup)
    global compiled_graph

    # Initialize PostgreSQL checkpointer
    # Adjust URL for asyncpg if needed (create_pool usually handles postgres://)
    # asyncpg usually expects postgresql://
    # Note: app/database.py converts to postgresql+asyncpg:// for SQLAlchemy,
    # but asyncpg directly uses postgresql://.
    pg_url = DATABASE_URL
    if "postgresql+asyncpg://" in pg_url:
            pg_url = pg_url.replace("postgresql+asyncpg://", "postgresql://")

    # Compile LangGraph (expensive operation - do once at startup)
    # conn = await asyncpg.connect(pg_url)
    conn = await psycopg.AsyncConnection.connect(DATABASE_URL)
    checkpointer = AsyncPostgresSaver(conn)
    compiled_graph = create_graph(checkpointer)
    print("✅ LangGraph compiled with PostgreSQL checkpointer")
    yield

    # Shutdown: Cleanup if needed
    print("👋 Shutting down Aura AI server...")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Aura AI",
    description="Agentic shopping assistant API",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints
@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {"status": "healthy", "service": "Aura AI", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "graph_compiled": compiled_graph is not None}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for conversing with the Aura AI assistant.

    Args:
        request: ChatRequest with message, user_id, and optional thread_id

    Returns:
        ChatResponse with assistant's response and conversation metadata
    """
    if compiled_graph is None:
        raise HTTPException(
            status_code=503,
            detail="Graph not initialized. Server may still be starting up.",
        )

    try:
        # Generate thread_id if not provided
        thread_id = request.thread_id or f"thread_{uuid.uuid4().hex[:8]}"

        # Prepare input for graph
        input_data = {
            "messages": [HumanMessage(content=request.message)],
        }

        # Configuration with user and thread IDs
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": request.user_id,
            }
        }

        # Invoke the graph asynchronously
        result = await compiled_graph.ainvoke(input_data, config)

        # Extract response from messages
        last_message = result["messages"][-1]
        response_text = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

        return ChatResponse(
            response=response_text,
            thread_id=thread_id,
            user_id=request.user_id,
            merged_images=result.get("merged_images", []),
        )

    except Exception as e:
        print(f"Error processing chat request: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
    )
