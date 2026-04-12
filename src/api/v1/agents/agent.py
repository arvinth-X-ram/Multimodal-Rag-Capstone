import os
from typing import TypedDict, List, Annotated
from pydantic import BaseModel, Field

import cohere
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.runnables.graph import MermaidDrawMethod

from src.api.v1.schemas.query_schema import AIResponse
from src.api.v1.tools.vector_search_tool import vector_search
from src.api.v1.tools.fts_search_tool import fts_search
from src.api.v1.tools.hybrid_search_tool import hybrid_search

os.environ["PYPPETEER_CHROMIUM_REVISION"] = "1263111"

load_dotenv(override=True)


# ─────────────────────────────────────────────────────────────────────────────
# State
# Each chunk in the lists is a plain dict:
#   { content, chunk_type, metadata, image_base64, page_number }
# ─────────────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    query: str
    retrieved_docs: list[dict]
    reranked_docs: list[dict]
    response: dict
    is_valid: bool
    attempts: int
    hallucination_attempts: int  # tracks how many times we've regenerated due to hallucination
    hallucination_grounded: bool  # True = grounded (no hallucination), False = hallucinating


# ─────────────────────────────────────────────────────────────────────────────
# LangChain @tool wrappers (used only so the LLM can bind & choose them)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def vector_search_tool(query: Annotated[str, "User query"]) -> list[dict]:
    """Semantic vector search — best for concepts, explanations, policies."""
    return vector_search(query, k=10)


@tool
def fts_search_tool(query: Annotated[str, "User query"]) -> list[dict]:
    """Full-text keyword search — best for exact terms, codes, names, dates."""
    return fts_search(query, k=10)


@tool
def hybrid_search_tool(query: Annotated[str, "User query"]) -> list[dict]:
    """Hybrid search (vector + FTS via RRF) — best default for most questions."""
    return hybrid_search(query, k=10)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text(content) -> str:
    """
    Safely extract a plain string from whatever shape the LLM returns.
    Gemini may return a list of content parts like:
      [{'type': 'text', 'text': 'yes', ...}]
    This helper handles that, plain strings, and single-dict cases.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(content, dict):
        return content.get("text", str(content))
    return str(content)


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────

def _build_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL", "gemini-3.1-pro-preview"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,
    )


def agent_node(state: RAGState) -> RAGState:
    """LLM picks which search tool to call, then executes it."""
    llm = _build_llm(temperature=0.0)
    tools = [vector_search_tool, fts_search_tool, hybrid_search_tool]
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = (
        "You are a document retrieval assistant. "
        "You MUST call exactly ONE search tool:\n"
        "- vector_search_tool  → concepts, explanations, general policy questions\n"
        "- fts_search_tool     → exact terms, clause codes, names, dates, numbers\n"
        "- hybrid_search_tool  → recommended default for most questions\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}"),
    ])

    response = (prompt | llm_with_tools).invoke({"query": state["query"]})

    _TOOL_MAP = {
        "vector_search_tool": vector_search,
        "fts_search_tool": fts_search,
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

    return {**state, "retrieved_docs": docs}


def rerank_node(state: RAGState) -> RAGState:
    """Cohere reranker over the retrieved chunks."""
    docs = state["retrieved_docs"]
    if not docs:
        return {**state, "reranked_docs": []}

    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

    # Cohere expects plain strings; use the content field
    rerank_response = co.rerank(
        model="rerank-english-v3.0",
        query=state["query"],
        documents=[doc["content"] for doc in docs],
        top_n=5,
    )

    reranked_docs = [docs[r.index] for r in rerank_response.results]
    print(f"[rerank] Top {len(reranked_docs)} chunks after reranking")
    return {**state, "reranked_docs": reranked_docs}


def validate_node(state: RAGState) -> RAGState:
    """Ask the LLM whether the retrieved chunks are sufficient."""
    llm = _build_llm(temperature=0.0)

    context_preview = "\n\n".join(
        f"Chunk {i+1}: {doc['content'][:280]}..."
        for i, doc in enumerate(state["reranked_docs"])
    )

    prompt = (
        f'User query: "{state["query"]}"\n\n'
        f"Retrieved & reranked chunks:\n{context_preview}\n\n"
        "Are these chunks sufficient to answer the query completely?\n"
        "Reply with ONLY one word: yes or no"
    )

    response = llm.invoke(prompt)
    is_valid = _extract_text(response.content).strip().lower().startswith("yes")

    print(f"[validate] Chunks are {'SUFFICIENT ✓' if is_valid else 'INSUFFICIENT ✗'}")
    return {**state, "is_valid": is_valid, "attempts": state.get("attempts", 0) + 1}


def rewrite_query_node(state: RAGState) -> RAGState:
    """Rewrite the query to improve retrieval on the next attempt."""
    llm = _build_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_template(
        "Rewrite the query to be clearer, more specific and better for search.\n"
        "Original: {query}\n\n"
        "Rewritten query (only the question, no explanation):"
    )

    result = (prompt | llm).invoke({"query": state["query"]})
    new_query = _extract_text(result.content).strip()

    print(f"[rewrite] Old → {state['query']}")
    print(f"[rewrite] New → {new_query}")

    return {
        **state,
        "query": new_query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "is_valid": False,
    }


def generate_answer_node(state: RAGState) -> RAGState:
    """Generate a structured answer from the reranked chunks."""
    llm = _build_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(AIResponse)

    # Collect the first image chunk's path (if any) — stored on the local filesystem.
    # We inject the path rather than the bytes so the response stays lightweight.
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
        ("system", "Answer ONLY using the provided context. Always cite source, page number and section."),
        ("human", "Context:\n{context}\n\nQuestion: {query}"),
    ])

    result = (prompt | structured_llm).invoke({"context": context, "query": state["query"]})
    
    # Inject the image path directly — never echo file bytes through the LLM
    response_dict = result.model_dump()
    response_dict["image_path"] = first_image_path
    # Keep image_base64 absent/None so the schema stays consistent
    response_dict.pop("image_base64", None)
    
    print("[generate_answer] Final answer generated")
    return {**state, "response": response_dict}


# ─────────────────────────────────────────────────────────────────────────────
# Hallucination Evaluator — replicates LangSmith langchain-ai/hallucination-eval
# ─────────────────────────────────────────────────────────────────────────────

class HallucinationGrade(BaseModel):
    """Structured output schema for the hallucination grader LLM judge."""
    score: str = Field(
        description="Binary 'yes' or 'no'. 'yes' means the answer is grounded in the facts; 'no' means it hallucinated."
    )


def hallucination_node(state: RAGState) -> RAGState:
    """
    LLM-as-a-Judge hallucination evaluator.

    Replicates the LangSmith `langchain-ai/hallucination-eval` prompt:
      - System: You are a grader assessing whether an LLM generation is grounded
                in / supported by a set of retrieved facts.
                Give a binary score 'yes' or 'no'. 'yes' means the answer IS
                grounded in / supported by the set of facts.
      - Human:  Set of facts: <documents>\nLLM generation: <generation>

    score='yes'  → answer is grounded,  proceed to END.
    score='no'   → hallucination detected, loop back to generate_answer.
    """
    llm = _build_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(HallucinationGrade)

    # Build the flat fact context from reranked docs (same documents used for generation)
    documents = "\n\n".join(
        f"Fact {i+1}: {doc['content']}"
        for i, doc in enumerate(state["reranked_docs"])
    )
    generation = state["response"].get("answer", "")

    # ── Exact LangSmith hallucination-eval prompt ──────────────────────────────
    system_prompt = (
        "You are a grader assessing whether an LLM generation is grounded in "
        "/ supported by a set of retrieved facts. \n"
        "Give a binary score 'yes' or 'no'. 'yes' means that the answer is "
        "grounded in / supported by the set of facts."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ])
    # ──────────────────────────────────────────────────────────────────────────

    result: HallucinationGrade = (prompt | structured_llm).invoke(
        {"documents": documents, "generation": generation}
    )

    score = result.score.strip().lower()
    is_grounded = score.startswith("yes")
    current_attempts = state.get("hallucination_attempts", 0) + 1

    if is_grounded:
        print(f"[hallucination_check] ✅ Grounded — score='{result.score}' (attempt {current_attempts})")
    else:
        print(f"[hallucination_check] 🚨 Hallucination detected — score='{result.score}' (attempt {current_attempts})")

    return {
        **state,
        "hallucination_grounded": is_grounded,
        "hallucination_attempts": current_attempts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_validate(state: RAGState) -> str:
    if state["is_valid"] or state.get("attempts", 0) >= 3:
        return "generate_answer"
    return "rewrite_query"


def _route_after_hallucination_check(state: RAGState) -> str:
    """
    Pure router — reads `hallucination_grounded` stored by hallucination_node.
    Mirrors LangSmith hallucination-eval routing:
      - grounded=True  → END
      - grounded=False → generate_answer (retry, capped at 2 hallucination_attempts)
    """
    is_grounded = state.get("hallucination_grounded", True)
    ha = state.get("hallucination_attempts", 0)

    if is_grounded:
        print("[hallucination_router] ✅ Grounded → END")
        return "end"
    elif ha >= 2:
        print(f"[hallucination_router] 🚨 Still hallucinating after {ha} attempt(s) — forcing END")
        return "end"
    else:
        print(f"[hallucination_router] 🔁 Hallucinating → regenerating answer (attempt {ha + 1})")
        return "generate_answer"


def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("agent", agent_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("validate", validate_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("hallucination_check", hallucination_node)

    graph.set_entry_point("agent")
    graph.add_edge("agent", "rerank")
    graph.add_edge("rerank", "validate")

    graph.add_conditional_edges(
        "validate",
        _route_after_validate,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
        },
    )

    graph.add_edge("rewrite_query", "agent")
    graph.add_edge("generate_answer", "hallucination_check")

    graph.add_conditional_edges(
        "hallucination_check",
        _route_after_hallucination_check,
        {
            "end": END,
            "generate_answer": "generate_answer",
        },
    )

    return graph.compile()


rag_graph = build_rag_graph()
# graph_image = rag_graph.get_graph().draw_mermaid_png(
#     draw_method=MermaidDrawMethod.PYPPETEER
# )
# with open("diagram\\rag_workflow.png","wb") as f:
#     f.write(graph_image)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_rag_agent(query: str) -> dict:
    """Run the full RAG pipeline and return the structured response dict."""
    initial_state: RAGState = {
        "query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
        "is_valid": False,
        "attempts": 0,
        "hallucination_attempts": 0,
        "hallucination_grounded": True,   # default; overwritten by hallucination_node
    }
    final_state = rag_graph.invoke(initial_state)
    return final_state["response"]