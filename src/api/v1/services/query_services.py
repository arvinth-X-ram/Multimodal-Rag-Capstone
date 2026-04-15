import json
from src.api.v1.agents.agent import build_rag_graph, run_rag_agent

# Initialize the compiled graph once so it can be reused efficiently
rag_graph = build_rag_graph()


def query_documents(q: str) -> dict:
    """
    Standard synchronous execution.
    Waits for the entire graph to finish and returns the final dictionary.
    """
    return run_rag_agent(query=q)
