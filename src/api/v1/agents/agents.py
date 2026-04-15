import os
import operator
from typing import Literal, TypedDict, List, Annotated, Optional
from pydantic import BaseModel, Field

import cohere
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from src.api.v1.schema.query_schema import AIResponse
from src.api.v1.tools.vector_search_tool import vector_search
from src.api.v1.tools.fts_search_tool import fts_search
from src.api.v1.tools.hybrid_search_tool import hybrid_search
from src.core.db import get_sql_database

load_dotenv(override=True)

# ─────────────────────────────────────────────────────────────────────────────
# State 
# ─────────────────────────────────────────────────────────────────────────────

def merge_dicts(existing: dict, new: dict) -> dict:
    return {**existing, **new}

def _extract_text(content) -> str:
    if isinstance(content, str): return content
    if isinstance(content, list):
        return "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content])
    return str(content)

class RAGState(TypedDict):
    query: Annotated[str, lambda x, y: y] 
    retrieved_docs: Annotated[list[dict], operator.add] 
    reranked_docs: Annotated[list[dict], lambda x, y: y]
    sql_raw_response: Annotated[Optional[dict], lambda x, y: y]
    doc_raw_response: Annotated[Optional[dict], lambda x, y: y]
    response: Annotated[dict, merge_dicts]
    paths_completed: Annotated[list[str], operator.add] 
    routes: Annotated[List[str], lambda x, y: y]
    is_valid: Annotated[bool, lambda x, y: y]
    attempts: Annotated[int, operator.add]
    hallucination_attempts: Annotated[int, operator.add]
    hallucination_grounded: Annotated[bool, lambda x, y: y]

class _RouteDecision(BaseModel):
    routes: List[Literal["credit_card_db", "document"]]
    reason: str

# ─────────────────────────────────────────────────────────────────────────────
# LangChain @tool wrappers (used only so the LLM can bind & choose them)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def vector_search_tool(query: Annotated[str, "User query"]) -> list[dict]:
    """Semantic vector search — best for concepts, explanations, policies."""
    return vector_search(query, k=5)


# @tool
# def fts_search_tool(query: Annotated[str, "User query"]) -> list[dict]:
#     """Full-text keyword search — best for exact terms, codes, names, dates."""
#     return fts_search(query, k=5)


@tool
def hybrid_search_tool(query: Annotated[str, "User query"]) -> list[dict]:
    """Hybrid search (vector + FTS via RRF) — best default for most questions."""
    return hybrid_search(query, k=5)

# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────

def _build_llm(temperature: float = 0.0):
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL", "gemini-1.5-pro"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,
    )

def router_node(state: RAGState) -> dict:
    llm = _build_llm()
    structured_llm = llm.with_structured_output(_RouteDecision)
    prompt = ChatPromptTemplate.from_messages([
        ("system",""" **Role**: You are a routing specialist for the NorthStar Bank Credit Card Spend Summarizer system. Your task is to analyze the user's query and determine the appropriate data source(s) required to answer it.

**Path Definitions**:
1. **'credit_card_db'**: Choose this path if the query requires specific data from the database tables provided (Customers, Credit Cards, Transactions, Reward Transactions, or Billing Statements). This includes requests for spend summaries, balance checks, transaction history, or specific customer account details.
2. **'document'**: Choose this path if the query asks for general information, policies, or explanations regarding NorthStar Bank credit card features, spend analysis logic, billing cycle rules, reward program terms, credit limit policies, or customer communication standards.

**Decision Rules**:
- If the query references specific account IDs (e.g., 'CC-881001'), transaction dates, or requests a calculation of spend, output ONLY: **credit_card_db**
- If the query asks about general bank procedures (e.g., "How is the minimum due calculated?" or "What are the benefits of the Gold variant?"), output ONLY: **document**
- If the query requires both specific data AND an explanation of a policy (e.g., "Show my March transactions and explain how my rewards were calculated"), output BOTH: **credit_card_db, document**

**Constraint**: Your output must contain ONLY the label(s) 'credit_card_db', 'document', or both. Do not provide conversational text."""),
        ("human", "{query}")
    ])
    decision = (prompt | structured_llm).invoke({"query": state["query"]})
    print(f"[router] Routes: {decision.routes}")
    return {"routes": decision.routes, "paths_completed": []}

def nl2sql_node(state: RAGState) -> dict:
    llm = _build_llm()
    db = get_sql_database()
    schema_info = db.get_table_info()

    sql_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a PostgreSQL expert. Return ONLY raw SQL based on schema: {schema}"),
        ("human", "Question: {question}")
    ])

    raw_sql = (sql_prompt | llm).invoke({"schema": schema_info, "question": state["query"]})
    content = _extract_text(raw_sql.content)
    generated_sql = content.strip().strip("```").replace("sql", "").strip()

    try:
        sql_result = db.run(generated_sql)
    except Exception as exc:
        sql_result = f"Error: {exc}"

    structured_llm = llm.with_structured_output(AIResponse)
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer concisely using SQL results."),
        ("human", "Query: {query}\nSQL: {sql}\nResults: {result}")
    ])

    answer = (answer_prompt | structured_llm).invoke({
        "query": state["query"], "sql": generated_sql, "result": sql_result
    })
    
    print("[nl2sql] Finished path")
    return {
        "sql_raw_response": {"answer": answer.answer, "document_name": "agentic_rag_db"}, 
        "paths_completed": ["credit_card_db"]
    }

def agent_node(state: RAGState) -> dict:
    """LLM picks which search tool to call, then executes it."""
    llm = _build_llm(temperature=0.0)
    # tools = [vector_search_tool, fts_search_tool, hybrid_search_tool]
    tools = [vector_search_tool, hybrid_search_tool]
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = (
        "You are a document retrieval assistant. "
        "You MUST call exactly ONE search tool:\n"
        "- vector_search_tool  → concepts, explanations, general policy questions\n"
        # "- fts_search_tool     → exact terms, clause codes, names, dates, numbers\n"
        "- hybrid_search_tool  → recommended default for most questions\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}"),
    ])

    response = (prompt | llm_with_tools).invoke({"query": state["query"]})

    # Inside agent_node
    _TOOL_MAP = {
        "vector_search_tool": vector_search,
        # "fts_search_tool": fts_search,
        "hybrid_search_tool": hybrid_search,
    }

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_query = tool_call["args"].get("query", state["query"])
        print(f"[agent] Calling → {tool_name}(query={tool_query!r})")
        docs = _TOOL_MAP.get(tool_name, hybrid_search)(tool_query, k=10)
    else:
        # Safety fallback
        print("[agent] No tool called → fallback to hybrid_search")
        docs = hybrid_search(state["query"], k=10)

    # print(f"[agent] Retrieved {docs}")
    return {**state, "retrieved_docs": docs}

def rerank_node(state: RAGState) -> dict:
    docs = state["retrieved_docs"]
    if not docs: return {"reranked_docs": []}
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    res = co.rerank(model="rerank-english-v3.0", query=state["query"], documents=[d["content"] for d in docs], top_n=5)
    return {"reranked_docs": [docs[r.index] for r in res.results]}

def validate_node(state: RAGState) -> dict:
    llm = _build_llm(temperature=0.0)
    context_preview = "\n\n".join([f"Chunk {i+1}: {doc['content'][:280]}..." for i, doc in enumerate(state["reranked_docs"])])
    prompt = (f"Query: {state['query']}\n\nContext:\n{context_preview}\n\n"
              "Is this sufficient to answer? And please note that the query may contain queries related to both documents and credit_card_dbs.Make sure the generated answer is relevant to the document part of the query, Even One chunck of data is enough. Reply ONLY 'yes' or 'no'.")
    
    response = llm.invoke(prompt)
    is_valid = _extract_text(response.content).strip().lower().startswith("yes")
    print(f"[validate] Sufficient: {is_valid}")
    return {"is_valid": is_valid, "attempts": 1}

def rewrite_query_node(state: RAGState) -> dict:
    llm = _build_llm(temperature=0.3)
    prompt = f"Rewrite this query for better search retrieval: {state['query']}"
    result = llm.invoke(prompt)
    new_query = _extract_text(result.content).strip()
    print(f"[rewrite] Old: {state['query']} -> New: {new_query}")
    return {"query": new_query, "retrieved_docs": [], "reranked_docs": []}

def generate_answer_node(state: RAGState) -> dict:
    llm = _build_llm()
    structured_llm = llm.with_structured_output(AIResponse)

    first_image_path: str | None = None
    for doc in state["reranked_docs"]:
        if doc.get("chunk_type") == "image" and doc.get("image_path"):
            first_image_path = doc["image_path"]
            print(f"[generate_answer] Image chunk found → {first_image_path}")
            break

    # Build context string; include image flag so the LLM knows multimodal data exists
    context_parts = []
    for doc in state["reranked_docs"]:
        meta = doc.get("metadata", {})
        source = meta.get("source") or doc.get("page_number", "?")
        page = meta.get("page") or doc.get("page_number", "?")
        section = meta.get("section", "")
        img_flag = " [contains image/figure]" if doc.get("image_path") else ""
        header = f"[Source: {source} | Page: {page} | Section: {section}]{img_flag}"
        context_parts.append(f"{header}\n{doc['content']}")

    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer ONLY using provided context. Cite sources."),
        ("human", "Context:\n{context}\n\nQuestion: {query}"),
    ])

    result = (prompt | structured_llm).invoke({"context": context, "query": state["query"]})
    # print(result)
    print("[generate_answer] Finished path")
    return {
        "doc_raw_response": {"answer": result, "document_name": "Policy PDF"}, 
        "paths_completed": ["document"]
    }

def aggregator_node(state: RAGState) -> dict:
    routes = state.get("routes", [])
    completed = state.get("paths_completed", [])
    
    if not all(r in completed for r in routes):
        print(f"[aggregator] Waiting... Completed: {completed} Needs: {routes}")
        return {} 

    print("[aggregator] ALL PATHS JOINED. Merging...")
    sql_data = state.get("sql_raw_response")
    doc_data = state.get("doc_raw_response")
    
    if sql_data and doc_data:
        llm = _build_llm()
        structured_llm = llm.with_structured_output(AIResponse)
        combined = structured_llm.invoke(f"Merge these answers cohesively and provide only one final answer to display in UI and do not ignore the Citation and Page number for document related data:\n1. {sql_data['answer']}\n2. {doc_data['answer']}")
        return {"response": {"answer": combined.model_dump()}}
    
    return {"response": sql_data or doc_data or {"answer": "No results found."}}

def hallucination_node(state: RAGState) -> dict:
    llm = _build_llm()
    structured_llm = llm.with_structured_output(HallucinationGrade)
    facts = "\n".join([d['content'] for d in state.get("reranked_docs", [])])
    gen = state["response"].get("answer", "")
    
    res = structured_llm.invoke(f"Facts: {facts}\nGeneration: {gen}")
    is_grounded = _extract_text(res.score).lower() == "yes"
    print(f"[hallucination] Grounded: {is_grounded}")
    return {"hallucination_grounded": is_grounded, "hallucination_attempts": 1}

class HallucinationGrade(BaseModel):
    score: str = Field(description="'yes' or 'no'")

# ─────────────────────────────────────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_validate(state: RAGState) -> str:
    if state["is_valid"] or state.get("attempts", 0) >= 3:
        return "generate_answer"
    return "rewrite_query"

def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("router", router_node)
    graph.add_node("nl2sql", nl2sql_node)
    graph.add_node("agent", agent_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("validate", validate_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("aggregator", aggregator_node)
    graph.add_node("hallucination_check", hallucination_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges("router", lambda state: state["routes"], {
        "credit_card_db": "nl2sql",
        "document": "agent"
    })

    # credit_card_db Path
    graph.add_edge("nl2sql", "aggregator")

    # Document Path with Sufficiency Loop
    graph.add_edge("agent", "rerank")
    graph.add_edge("rerank", "validate")
    graph.add_conditional_edges("validate", _route_after_validate, {
        "generate_answer": "generate_answer",
        "rewrite_query": "rewrite_query"
    })
    graph.add_edge("rewrite_query", "agent")
    graph.add_edge("generate_answer", "aggregator")

    # Convergence
    graph.add_edge("aggregator", "hallucination_check")
    graph.add_conditional_edges("hallucination_check", 
        lambda state: "end" if state["hallucination_grounded"] or state["hallucination_attempts"] >= 2 else "generate_answer",
        {"end": END, "generate_answer": "generate_answer"}
    )

    return graph.compile()

rag_graph = build_rag_graph()

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_rag_agent(query: str) -> dict:
    initial_state = {
        "query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
        "sql_raw_response": None,
        "doc_raw_response": None,
        "paths_completed": [],
        "routes": [],
        "is_valid": False,
        "attempts": 0,
        "hallucination_attempts": 0,
        "hallucination_grounded": True,
    }
    final_state = rag_graph.invoke(initial_state)
    return final_state["response"]