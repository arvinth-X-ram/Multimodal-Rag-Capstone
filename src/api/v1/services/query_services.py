from src.api.v1.tools.vector_search_tool import vector_search
from src.api.v1.agents.agents import run_rag_agent
def query_documents(q:str)->dict:
    return run_rag_agent(query=q)