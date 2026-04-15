import os
from typing import Literal, TypedDict, Annotated
from pydantic import BaseModel, Field
import cohere
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# Adjust these imports based on your exact project structure
from src.api.v1.schemas.query_schema import AIResponse
from src.api.v1.tools.vector_search_tool import vector_search
from src.api.v1.tools.fts_search_tool import fts_search
from src.api.v1.tools.hybrid_search_tool import hybrid_search
from src.core.db import get_sql_database

os.environ["PYPPETEER_CHROMIUM_REVISION"] = "1263111"
load_dotenv(override=True)

# ─────────────────────────────────────────────────────────────────────────────
# State & Routing Schema
# ─────────────────────────────────────────────────────────────────────────────
class RAGState(TypedDict):
    query: str
    sql_query: str  
    rag_query: str  
    retrieved_docs: list[dict]
    reranked_docs: list[dict]
    response: dict
    is_valid: bool
    attempts: int
    route: str  
    hallucination_attempts: int
    hallucination_grounded: bool
    sql_response: dict | None
    rag_response: dict | None

class _RouteDecision(BaseModel):
    route: Literal["product", "document", "both"]
    reason: str
    sql_query: str = Field(description="The exact part of the query meant for the SQL database.")
    rag_query: str = Field(description="The exact part of the query meant for document search.")

# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────
@tool
def fts_search_tool(query: Annotated[str, "User query"]) -> list[dict]:
    """Full-text keyword search — best for exact terms, codes, names, dates."""
    return fts_search(query, k=10)

@tool
def vector_search_tool(query: Annotated[str, "User query"]) -> list[dict]:
    """Semantic vector search — best for concepts, explanations, policies."""
    return vector_search(query, k=5)

@tool
def hybrid_search_tool(query: Annotated[str, "User query"]) -> list[dict]:
    """Hybrid search (vector + FTS via RRF) — best default for most questions."""
    return hybrid_search(query, k=5)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _extract_text(content) -> str:
    if isinstance(content, str): return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    if isinstance(content, dict): return content.get("text", str(content))
    return str(content)

def _build_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL", "gemini-3.1-pro-preview"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────
def nlsql_router_node(state: RAGState) -> dict:
    print(f"\n[NODE: ROUTER] Analyzing user query: '{state['query']}'")
    llm = _build_llm()
    structured_llm = llm.with_structured_output(_RouteDecision)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query router for an agentic RAG system.
Classify the user's query into EXACTLY one of three routes:
"product"  — ONLY about products, prices, stock, categories, orders.
"document" — ONLY about policies, procedures, financial reports, unstructured text.
"both"     — requires information from BOTH sources.

If route is "both", split it into 'sql_query' and 'rag_query'."""),
        ("human", "Query: {query}")
    ])
    decision = (prompt | structured_llm).invoke({"query": state["query"]})
    print(f"  -> Decision: Routed to '{decision.route}'")
    if decision.route == "both":
        print(f"  -> SQL Splitted Query: '{decision.sql_query}'")
        print(f"  -> RAG Splitted Query: '{decision.rag_query}'")
        
    return {
        "route": decision.route,
        "sql_query": decision.sql_query if decision.sql_query.strip() else state["query"],
        "rag_query": decision.rag_query if decision.rag_query.strip() else state["query"],
    }

def nl2sql_node(state: RAGState) -> dict:
    target_query = state.get("sql_query") or state["query"]
    print(f"\n[NODE: NL2SQL] Translating query to SQL for target: '{target_query}'")
    
    llm = _build_llm()
    db = get_sql_database()
    schema_info = db.get_table_info()
    
    sql_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a PostgreSQL expert. Return ONLY raw SQL based on the schema:\n{schema}"),
        ("human", "Question: {question}")
    ])
    raw_sql = (sql_prompt | llm).invoke({"schema": schema_info, "question": target_query})
    generated_sql = _extract_text(raw_sql.content).strip().strip("```").strip()
    if generated_sql.lower().startswith("sql"): generated_sql = generated_sql[3:].strip()

    print(f"  -> Generated SQL: {generated_sql}")

    try:
        sql_result = db.run(generated_sql)
        print(f"  -> SQL Execution: SUCCESS (Result length: {len(str(sql_result))})")
    except Exception as exc:
        sql_result = f"SQL execution error: {exc}"
        print(f"  -> SQL Execution: ERROR - {exc}")

    structured_llm = llm.with_structured_output(AIResponse)
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Concise data analyst. Use ONLY SQL results. Set page_no='N/A', document_name='agentic_rag_db', policy_citations='N/A'."),
        ("human", "Question: {query}\n\nSQL: {sql}\n\nResults: {result}")
    ])
    answer = (answer_prompt | structured_llm).invoke({"query": target_query, "sql": generated_sql, "result": sql_result})
    
    response = answer.model_dump()
    return {"sql_response": response, "response": response}

def agent_node(state: RAGState) -> dict:
    target_query = state.get("rag_query") or state["query"]
    print(f"\n[NODE: AGENT] Determining retrieval tool for query: '{target_query}'")
    
    llm = _build_llm()
    tools = [fts_search_tool, vector_search_tool, hybrid_search_tool]
    llm_with_tools = llm.bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Document retrieval assistant. Call exactly ONE search tool."),
        ("human", "{query}"),
    ])
    response = (prompt | llm_with_tools).invoke({"query": target_query})

    _TOOL_MAP = {"fts_search_tool": fts_search, "vector_search_tool": vector_search, "hybrid_search_tool": hybrid_search}
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        print(f"  -> Selected Tool: {tool_call['name']}")
        docs = _TOOL_MAP.get(tool_call["name"], hybrid_search)(tool_call["args"].get("query", target_query), k=10)
    else:
        print(f"  -> No specific tool selected, defaulting to: hybrid_search")
        docs = hybrid_search(target_query, k=10)
        
    print(f"  -> Retrieved {len(docs)} documents.")
    return {"retrieved_docs": docs}

def rerank_node(state: RAGState) -> dict:
    docs = state["retrieved_docs"]
    print(f"\n[NODE: RERANK] Reranking {len(docs)} retrieved documents using Cohere")
    if not docs: 
        print("  -> No documents to rerank.")
        return {"reranked_docs": []}
        
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    rerank_response = co.rerank(model="rerank-english-v3.0", query=state.get("rag_query") or state["query"], documents=[doc["content"] for doc in docs], top_n=5)
    reranked = [docs[r.index] for r in rerank_response.results]
    print(f"  -> Reranking complete. Kept top {len(reranked)} documents.")
    return {"reranked_docs": reranked}

def validate_node(state: RAGState) -> dict:
    attempts = state.get("attempts", 0) + 1
    print(f"\n[NODE: VALIDATE] Checking relevance of documents (Attempt {attempts}/3)")
    llm = _build_llm()
    context = "\n\n".join(f"Chunk {i+1}: {doc['content'][:280]}" for i, doc in enumerate(state["reranked_docs"]))
    prompt = f'Query: "{state.get("rag_query") or state["query"]}"\n\nChunks:\n{context}\n\nSufficient? yes/no'
    is_valid = _extract_text(llm.invoke(prompt).content).strip().lower().startswith("yes")
    print(f"  -> Sufficient context found? {is_valid}")
    return {"is_valid": is_valid, "attempts": attempts}

def rewrite_query_node(state: RAGState) -> dict:
    print(f"\n[NODE: REWRITE] Context insufficient. Rewriting query for better retrieval.")
    llm = _build_llm(temperature=0.3)
    prompt = ChatPromptTemplate.from_template("Rewrite for doc search: {query}")
    result = (prompt | llm).invoke({"query": state.get("rag_query") or state["query"]})
    rewritten_query = _extract_text(result.content).strip()
    print(f"  -> Rewritten Query: '{rewritten_query}'")
    return {"rag_query": rewritten_query, "retrieved_docs": [], "reranked_docs": [], "is_valid": False}

def generate_answer_node(state: RAGState) -> dict:
    print(f"\n[NODE: GENERATE ANSWER] Synthesizing final answer from RAG context")
    llm = _build_llm()
    structured_llm = llm.with_structured_output(AIResponse)
    target_query = state.get("rag_query") or state["query"]
    
    first_image = next((doc["image_path"] for doc in state["reranked_docs"] if doc.get("chunk_type") == "image" and doc.get("image_path")), None)
    context = "\n\n".join(f"[Source: {doc.get('metadata',{}).get('source','?')} | Page: {doc.get('page_number', '?')}] {doc['content']}" for doc in state["reranked_docs"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer ONLY using context. You MUST fill the 'policy_citations', 'page_no', and 'document_name' fields based on the source headers provided in the context."),
        ("human", "Context:\n{context}\n\nQuestion: {query}"),
    ])
    result = (prompt | structured_llm).invoke({"context": context, "query": target_query})
    response_dict = result.model_dump()
    response_dict["image_path"] = first_image
    print(f"  -> Generation complete.")
    return {"rag_response": response_dict, "response": response_dict}

class HallucinationGrade(BaseModel):
    score: str = Field(description="Binary 'yes' or 'no'.")

def hallucination_node(state: RAGState) -> dict:
    attempts = state.get("hallucination_attempts", 0) + 1
    print(f"\n[NODE: HALLUCINATION CHECK] Verifying grounding against context (Attempt {attempts}/2)")
    llm = _build_llm()
    structured_llm = llm.with_structured_output(HallucinationGrade)
    documents = "\n\n".join(doc['content'] for doc in state["reranked_docs"])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Binary score 'yes' or 'no' for grounding."),
        ("human", "Facts: {documents}\n\nGeneration: {generation}"),
    ])
    result = (prompt | structured_llm).invoke({"documents": documents, "generation": state["response"].get("answer", "")})
    is_grounded = result.score.strip().lower().startswith("yes")
    print(f"  -> Is Grounded? {is_grounded}")
    return {"hallucination_grounded": is_grounded, "hallucination_attempts": attempts}

def both_start_node(state: RAGState) -> dict:
    print(f"\n[NODE: BOTH_START] Initiating parallel processing for BOTH SQL and RAG routes.")
    return {}

def merge_node(state: RAGState) -> dict:
    print(f"\n[NODE: MERGE] Combining outputs from SQL and RAG pipelines")
    sql, rag = state.get("sql_response") or {}, state.get("rag_response") or {}
    llm = _build_llm()
    structured_llm = llm.with_structured_output(AIResponse)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant.
Combine the two answers below into ONE cohesive, well-formatted response.
- Use information from BOTH sources.
- Preserve and use the metadata provided (citations, page numbers, document names).
- For the SQL part, the source is 'agentic_rag_db'.
- Keep the same AIResponse structure."""),
        ("human", """Original Combined Query: {query}

--- DATABASE DATA (SQL) ---
Answer: {sql_answer}

--- DOCUMENT DATA (RAG) ---
Answer: {rag_answer}
Citations: {rag_citations}
Page Number: {rag_page}
Document Name: {rag_doc_name}""")
    ])
    
    combined = (prompt | structured_llm).invoke({
        "query": state["query"], 
        "sql_answer": sql.get("answer", "No database info."), 
        "rag_answer": rag.get("answer", "No document info."),
        "rag_citations": rag.get("policy_citations", "N/A"),
        "rag_page": rag.get("page_no", "N/A"),
        "rag_doc_name": rag.get("document_name", "N/A"),
    })
    
    final_dict = combined.model_dump()
    final_dict["image_path"] = rag.get("image_path") or sql.get("image_path")
    print(f"  -> Merge complete.")
    return {"response": final_dict, "sql_response": sql, "rag_response": rag}

# ─────────────────────────────────────────────────────────────────────────────
# Edges & Compilation
# ─────────────────────────────────────────────────────────────────────────────
def _route_initial_query(state: RAGState) -> str:
    r = state.get("route", "")
    if r == "product": return "nl2sql"
    if r == "document": return "agent"
    return "both_start"

def route_after_nl2sql(state: RAGState) -> str:
    return "merge" if state.get("route") == "both" else END

def _route_after_validate(state: RAGState) -> str:
    return "generate_answer" if state["is_valid"] or state.get("attempts", 0) >= 3 else "rewrite_query"

def _route_after_hallucination_check(state: RAGState) -> str:
    if state.get("hallucination_grounded", True) or state.get("hallucination_attempts", 0) >= 2:
        return "merge" if state.get("route") == "both" else END
    return "generate_answer"

def build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("router", nlsql_router_node)
    graph.add_node("nl2sql", nl2sql_node)
    graph.add_node("agent", agent_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("validate", validate_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("hallucination_check", hallucination_node)
    graph.add_node("both_start", both_start_node)
    graph.add_node("merge", merge_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", _route_initial_query, {"nl2sql": "nl2sql", "agent": "agent", "both_start": "both_start"})
    graph.add_conditional_edges("nl2sql", route_after_nl2sql, {"merge": "merge", END: END})
    graph.add_edge("agent", "rerank")
    graph.add_edge("rerank", "validate")
    graph.add_conditional_edges("validate", _route_after_validate, {"generate_answer": "generate_answer", "rewrite_query": "rewrite_query"})
    graph.add_edge("rewrite_query", "agent")
    graph.add_edge("generate_answer", "hallucination_check")
    graph.add_conditional_edges("hallucination_check", _route_after_hallucination_check, {"merge": "merge", END: END, "generate_answer": "generate_answer"})
    graph.add_edge("both_start", "nl2sql")
    graph.add_edge("both_start", "agent")
    graph.add_edge("merge", END)
    return graph.compile()

def run_rag_agent(query: str) -> dict:
    print(f"\n========================================")
    print(f" STARTING LANGGRAPH AGENT EXECUTION")
    print(f"========================================")
    rag_graph = build_rag_graph()
    initial_state: RAGState = {
        "query": query, "sql_query": "", "rag_query": "", "retrieved_docs": [], "reranked_docs": [], "response": {},
        "is_valid": False, "attempts": 0, "route": "", "hallucination_attempts": 0, "hallucination_grounded": True,
        "sql_response": None, "rag_response": None
    }
    
    result = rag_graph.invoke(initial_state)["response"]
    print(f"\n========================================")
    print(f" AGENT EXECUTION COMPLETE")
    print(f"========================================\n")
    return result