import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.api.v1.tools.vector_search_tool import vector_search
from src.api.v1.tools.hybrid_search_tool import _hybrid_search
from src.api.v1.tools.fts_search_tool import fts_search
from langchain_core.tools import tool

from langchain.agents import create_agent 

load_dotenv(override=True)
sources = []

@tool
def vector_search_tool(query: str):
    """
    It is best suited for natural‑language and concept‑based questions.
    Use this tool when the user is seeking explanations, interpretations of policy intent, or descriptions of  processes and frameworks.
    Do not rely on exact keyword or acronym matching when using this tool; focus on conceptual relevance instead.
    """
    val = vector_search(query)
    print("The: \n",val)
    sources.append(val)
    return val

@tool
def fts_tool(query:str):
    """
    It is best suited for identifying specific  frameworks, models, acronyms, fixed terminology, document titles, and section headers.
    Use this tool when the user query contains precise or well‑defined  terms (e.g., PD, LGD, exposure limits, policy names).
    Do not use this tool for conversational, explanatory, or scenario‑based  questions.
   """
    val = fts_search(query)
    print("The: \n",val)
    sources.append(val)
    return val

@tool
def hybrid_search_tool(query: str):
    """
    Use this tool when the query requires both exact terminology matching and contextual understanding.
    It is best suited for long, complex, or ambiguous credit risk questions, including scenario‑based or decision‑oriented queries.
    This tool combines keyword‑based search with semantic similarity to retrieve the most relevant credit risk policies and rules.
    Use this tool when it is unclear whether a purely keyword or purely semantic search would be sufficient.
   """
    val = _hybrid_search(query)
    sources.append(val)
    return val

_llm = ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_LLM_MODEL"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    # tools = [search_tools]
)

def query_documents(query: str, k: int = 5, chunk_type: str | None = None) -> dict:

    my_agent = create_agent(
    model = _llm,
    tools = [hybrid_search_tool],
    system_prompt = """You are a helpful assistant for document question-answering. 
    Answer the question using Tools provide by searching in the knowledge Base. 
    If the answer is not present in the context, say you don't know. 
    When citing information, mention the page number and section.""")

    response = my_agent.invoke(
        {
        "messages": query
        },
        config=
        {
          "tags": ["CREDIT_RAG_AGENT"],
          "metadata": {
              "user_id": "user_001",
              "feature": "Can able to perform vector,fts and hybrid search to retrive doccuments.",
              "env": "dev"
                      },
          "run_name": "CREDIT_RAG_RUN"

        })


    print(response["messages"][-1].content)
    answer = response["messages"][-1].text
    return {
        "answer": answer,
        "sources": sources,
    }


