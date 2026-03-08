import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from sqlalchemy.orm import Session

from ..config.config import OPENAI_API_KEY
from .ai_analysis_memory_store import AiAnalysisMemory, load_memory, save_memory, upsert_knowledge
from .sap_sql_agent import run_sap_sql_agent, _serialize_value  # type: ignore
from .ai_chart_generator import analyze_visualization_needs, chart_specs_to_json
from .training_data_collector import log_query_execution, get_few_shot_examples
from .query_cache import find_similar_cached_query, cache_query_result

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    reply: str
    action: str
    reason: str = ""
    sql: str = ""
    rows_preview: Optional[List[Dict[str, Any]]] = None
    compare: Optional[Dict[str, Any]] = None
    memory_updated: bool = False
    charts: Optional[List[Dict[str, Any]]] = None


def _get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def _safe_json_extract(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}


def _decide_action(client: OpenAI, user_query: str, mem: AiAnalysisMemory) -> Tuple[str, str]:
    last_sql = (mem.last_sql or "")[:2000]
    last_q = (mem.last_user_query or "")[:500]
    prompt = f"""
You are assisting with SQL query generation for an analytics chat.

User query: "{user_query}"
Last user query in this chat: "{last_q}"
Last SQL query (if any): "{last_sql}"

Task:
Decide whether this query is:
1) identical to a query asked before (reuse)
2) a comparison between multiple datasets/periods (compare)
3) a follow-up that can be answered from prior results/context without new SQL (follow-up)
4) new knowledge/instructions the user wants remembered (knowledge)
5) a new query requiring new SQL and new data (new)

Return JSON only:
{{
  "action": "reuse"|"compare"|"follow-up"|"knowledge"|"new",
  "reason": "short reason"
}}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=120,
    )
    decision = _safe_json_extract((resp.choices[0].message.content or "").strip())
    action = str(decision.get("action") or "new").strip()
    reason = str(decision.get("reason") or "").strip()
    if action not in {"reuse", "compare", "follow-up", "knowledge", "new"}:
        action = "new"
    return action, reason


def _is_explicit_knowledge_instruction(user_query: str) -> bool:
    """
    Detect when the user is clearly giving an instruction to save/remember,
    so we always treat it as "knowledge" and do not run SQL or answer from context.
    """
    q = (user_query or "").strip().lower()
    if not q or len(q) < 10:
        return False
    prefixes = (
        "remember:",
        "remember that",
        "save this:",
        "save that",
        "store this:",
        "store that",
        "note:",
        "note that",
        "for future:",
        "when i ask about",
        "when i ask for",
    )
    return any(q.startswith(p) for p in prefixes)


def _is_implicit_knowledge_instruction(user_query: str) -> bool:
    """
    Detect instruction-like phrases that should be saved even without "Remember:".
    Examples: "always add period", "join these tables", "use EKPO for cost queries".
    """
    q = (user_query or "").strip().lower()
    if not q or len(q) < 8:
        return False
    # Instruction patterns (phrases that indicate a rule/preference to remember)
    patterns = (
        "always add",
        "always include",
        "add period",
        "include period",
        "add year",
        "include year",
        "add quantity",
        "include quantity",
        "join these tables",
        "use these tables",
        "use these for",
        "for cost",
        "for sales",
        "for vendor",
        "for invoice",
        "when i ask",
        "when asking about",
    )
    return any(p in q for p in patterns)


def _is_small_chitchat(user_query: str) -> bool:
    """
    Fast heuristic: detect trivial greetings/thanks that should NOT trigger SQL.
    """
    q = (user_query or "").strip().lower()
    if not q:
        return False
    # Single-word or very short chit-chat
    simple_greetings = {
        "hi",
        "hello",
        "hey",
        "yo",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "hola",
        "bye",
        "good morning",
        "good evening",
        "good night",
    }
    # If the query is very short and matches a greeting/thanks, treat as chit-chat
    if len(q.split()) <= 3 and any(g in q for g in simple_greetings):
        # Avoid false positives when there are clear data keywords
        data_keywords = ["invoice", "invoices", "sales", "revenue", "customer", "product", "country", "industry"]
        if not any(k in q for k in data_keywords):
            return True
    return False


def _rows_preview(rows: List[Dict[str, Any]], limit: int = 30) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows[:limit]:
        clean = {str(k): _serialize_value(v) for k, v in (r or {}).items()}
        out.append(clean)
    return out


def _split_compare_query(client: OpenAI, user_query: str) -> List[str]:
    """
    INVOICE_BOT uses split_comparison_query. We mimic via LLM:
    return 2+ sub-questions that can be executed separately.
    """
    prompt = f"""
User query: "{user_query}"

Task:
If the user is asking to compare two or more things (time periods, customers, products, countries),
rewrite into multiple standalone questions that can be answered by SQL independently.

Return JSON only:
{{
  "subqueries": ["...", "..."]
}}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=250,
    )
    data = _safe_json_extract((resp.choices[0].message.content or "").strip())
    subs = data.get("subqueries")
    if isinstance(subs, list):
        cleaned = [str(s).strip() for s in subs if str(s).strip()]
        return cleaned[:4]
    return []


def _compare_numeric(datasets: List[Tuple[str, List[Dict[str, Any]]]]) -> Dict[str, Any]:
    """
    Lightweight compare: for each dataset compute numeric column sums/means where possible.
    """
    def is_num(x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    summary: Dict[str, Any] = {"datasets": []}
    for label, rows in datasets:
        if not rows:
            summary["datasets"].append({"label": label, "row_count": 0, "numeric": {}})
            continue
        numeric_acc: Dict[str, Dict[str, float]] = {}
        counts: Dict[str, int] = {}
        for r in rows:
            if not isinstance(r, dict):
                continue
            for k, v in r.items():
                if is_num(v):
                    if k not in numeric_acc:
                        numeric_acc[k] = {"sum": 0.0}
                        counts[k] = 0
                    numeric_acc[k]["sum"] += float(v)
                    counts[k] += 1
        numeric_out: Dict[str, Dict[str, float]] = {}
        for k, agg in numeric_acc.items():
            c = counts.get(k, 0) or 0
            numeric_out[k] = {"sum": agg["sum"], "mean": (agg["sum"] / c) if c else 0.0}
        summary["datasets"].append({"label": label, "row_count": len(rows), "numeric": numeric_out})
    return summary


def run_ai_analysis_orchestrator(
    *,
    api_key: Optional[str],
    user_id: int,
    user_query: str,
    db: Session,
    conversation_history: Optional[list] = None,
    context_str: str = "",
    sap_db: Optional[Session] = None,
) -> OrchestratorResult:
    """
    Orchestrates INVOICE_BOT-like behaviors:
    - action decision (reuse/compare/follow-up/knowledge/new)
    - persisted memory per user (last SQL, last rows, knowledge)
    - runs sap_sql_agent when needed
    Returns a response payload that keeps `reply` for the current frontend.
    """
    effective_key = api_key or OPENAI_API_KEY
    if not effective_key:
        return OrchestratorResult(reply="AI analysis is not configured (missing OPENAI_API_KEY).", action="error", reason="missing_api_key")

    client = _get_client(effective_key)
    mem = load_memory(db, user_id)
    # If user gives an instruction to save (explicit or implicit), treat as knowledge (don't run SQL).
    if _is_explicit_knowledge_instruction(user_query):
        action, reason = "knowledge", "explicit_save_instruction"
    elif _is_implicit_knowledge_instruction(user_query):
        action, reason = "knowledge", "implicit_instruction_detected"
    else:
        action, reason = _decide_action(client, user_query, mem)

    # Pure chit-chat (greetings, thanks, etc.) – do NOT hit the database.
    if _is_small_chitchat(user_query):
        prompt = f"""
You are a friendly business assistant.

The user sent a short greeting or casual message that does NOT require database queries.
Respond briefly and naturally. Do NOT mention SQL or data, just be polite.

User: {user_query}
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=80,
        )
        reply = (resp.choices[0].message.content or "").strip()
        mem.last_user_query = user_query
        save_memory(db, mem)
        return OrchestratorResult(
            reply=reply or "Hello!",
            action="chitchat",
            reason="short_greeting_no_sql",
            sql="",
            rows_preview=None,
            memory_updated=True,
        )

    # knowledge: store a short instruction snippet
    if action == "knowledge":
        upsert_knowledge(db, user_id, f"note_{len(mem.knowledge())+1}", user_query.strip()[:2000])
        return OrchestratorResult(
            reply="Saved that as a note for this chat session. Ask me a question anytime and I’ll use it as context.",
            action=action,
            reason=reason,
            memory_updated=True,
        )

    # follow-up: respond from memory + context without new SQL
    if action == "follow-up":
        last_rows = mem.last_rows()
        prompt = f"""
You are a business analyst assistant. Answer the user's follow-up using prior context and the last result sample when relevant.

Dashboard context:
{context_str[:6000]}

Last SQL (if any):
{(mem.last_sql or '')[:3000]}

Last result sample (JSON, may be empty):
{json.dumps(last_rows[:10], default=str)[:6000]}

Conversation (latest last):
{json.dumps((conversation_history or [])[-10:], default=str)[:6000]}

User follow-up:
{user_query}

Answer concisely.
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=700,
        )
        return OrchestratorResult(
            reply=(resp.choices[0].message.content or "").strip() or "I couldn’t generate a response.",
            action=action,
            reason=reason,
            sql=mem.last_sql or "",
            rows_preview=last_rows[:30] if last_rows else None,
        )

    # compare: run multiple subqueries, compare numeric summaries, then summarize differences
    if action == "compare":
        subqueries = _split_compare_query(client, user_query)
        if len(subqueries) < 2:
            action = "new"
        else:
            sql_db = sap_db or db
            knowledge = mem.knowledge()
            knowledge_context = "\n".join(str(v) for v in knowledge.values()) if knowledge else None
            few_shot_compare = get_few_shot_examples(db, limit=3, user_id=user_id) or get_few_shot_examples(db, limit=3)
            datasets: List[Tuple[str, List[Dict[str, Any]]]] = []
            sqls: List[str] = []
            for sq in subqueries[:3]:
                r = run_sap_sql_agent(
                    sq, sql_db,
                    knowledge_context=knowledge_context,
                    few_shot_examples=few_shot_compare if few_shot_compare else None,
                )
                if not r or not r.rows:
                    datasets.append((sq, []))
                    sqls.append(r.sql if r else "")
                else:
                    datasets.append((sq, r.rows))
                    sqls.append(r.sql)
            compare_summary = _compare_numeric(datasets)
            prompt = f"""
You are a data analyst. The user asked for a comparison:
"{user_query}"

We executed these sub-questions:
{json.dumps(subqueries, indent=2)}

Comparison summary (auto-computed numeric sums/means):
{json.dumps(compare_summary, indent=2)}

Write a short comparison summary (5-10 sentences). Call out the biggest differences.
If data is missing for a dataset, mention it clearly.
"""
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=800,
            )
            reply = (resp.choices[0].message.content or "").strip()
            # update memory to last dataset (helps follow-ups)
            last_sql = next((s for s in reversed(sqls) if s), "")
            last_rows = next((rows for _, rows in reversed(datasets) if rows), [])
            
            # Generate comparison charts
            charts_data = None
            try:
                if last_rows:
                    chart_specs = analyze_visualization_needs(last_rows, user_query, "compare", last_sql)
                    if chart_specs:
                        charts_data = chart_specs_to_json(chart_specs)
                        logger.info(f"Generated {len(charts_data)} comparison chart(s)")
            except Exception as chart_err:
                logger.warning(f"Comparison chart generation failed: {chart_err}")
            
            mem.last_user_query = user_query
            mem.last_sql = last_sql
            mem.last_rows_json = json.dumps(_rows_preview(last_rows, limit=80), default=str)
            save_memory(db, mem)
            return OrchestratorResult(
                reply=reply or "Comparison complete, but I couldn’t generate a narrative summary.",
                action="compare",
                reason=reason,
                sql=last_sql,
                rows_preview=_rows_preview(last_rows) if last_rows else None,
                compare={"subqueries": subqueries, "sqls": sqls, "summary": compare_summary},
                memory_updated=True,
                charts=charts_data,
            )

    # reuse: re-run last SQL if we have it, otherwise treat as new
    if action == "reuse" and mem.last_sql:
        # In serverless, re-executing arbitrary SQL can be expensive/unreliable.
        # Prefer reusing the last cached rows preview for "reuse" answers; "new" will re-run via sap_sql_agent.
        rows_list = mem.last_rows()
        prompt = f"""
User asked: "{user_query}"

We are reusing the prior SQL:
```sql
{mem.last_sql}
```

Result preview JSON:
{json.dumps(_rows_preview(rows_list, limit=20), default=str)}

Explain the answer briefly (3-8 sentences). If result is empty, say so and suggest a refined question.
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=650,
        )
        reply = (resp.choices[0].message.content or "").strip()
        mem.last_user_query = user_query
        mem.last_rows_json = json.dumps(_rows_preview(rows_list, limit=80), default=str)
        save_memory(db, mem)
        return OrchestratorResult(
            reply=reply or "Reused the prior query, but no response was generated.",
            action="reuse",
            reason=reason,
            sql=mem.last_sql,
            rows_preview=_rows_preview(rows_list) if rows_list else None,
            memory_updated=True,
        )

    # new: run SAP SQL agent, store sql+rows and return summary.
    # Check cache first for similar queries
    try:
        cached_result = find_similar_cached_query(db, user_query, threshold=0.85)
        if cached_result:
            logger.info(f"Serving from cache (similarity={cached_result.get('similarity', 0):.3f})")
            return OrchestratorResult(
                reply=cached_result["result_summary"],
                action="cached",
                reason=f"cache_hit_similarity_{cached_result.get('similarity', 0):.2f}",
                sql=cached_result.get("sql_query", ""),
                rows_preview=cached_result.get("result_preview"),
                memory_updated=False,
                charts=cached_result.get("charts"),
            )
    except Exception as cache_err:
        logger.warning(f"Cache lookup failed: {cache_err}")
    
    # If the agent cannot produce useful rows, fall back to answering from the
    # already-built dashboard context only (which we know works and has data).
    sql_db = sap_db or db
    knowledge = mem.knowledge()
    knowledge_context = "\n".join(str(v) for v in knowledge.values()) if knowledge else None
    few_shot = get_few_shot_examples(db, limit=3, user_id=user_id) or get_few_shot_examples(db, limit=3)
    result = run_sap_sql_agent(
        user_query, sql_db,
        knowledge_context=knowledge_context,
        few_shot_examples=few_shot if few_shot else None,
    )
    if not result or not result.rows:
        # If user asked about costs and we have cost-table knowledge, explain that the query ran but returned no rows.
        q_lower = (user_query or "").lower()
        knowledge = mem.knowledge()
        knowledge_str = " ".join(str(v) for v in knowledge.values()).lower()
        cost_related = "cost" in q_lower or "costing" in q_lower or "vendor" in q_lower
        has_cost_tables_note = knowledge and ("ekpo" in knowledge_str or "rbkp" in knowledge_str or "rseg" in knowledge_str)
        if cost_related and has_cost_tables_note:
            fallback_reply = (
                "I used your cost tables (e.g. EKPO, RBKP, RSEG) for this question, but the query returned no rows. "
                "Those tables may be empty in the database for the current period, or the question may need different filters. "
                "You can try asking for costs by vendor, by material, or by purchase order; if data exists, I’ll show it."
            )
            mem.last_user_query = user_query
            save_memory(db, mem)
            return OrchestratorResult(
                reply=fallback_reply,
                action="new",
                reason=reason or "cost_query_no_rows",
                sql=result.sql if result else "",
                rows_preview=None,
                memory_updated=True,
            )
        # Fallback: answer from context_str alone, without relying on live SQL rows
        if context_str.strip():
            prompt = f"""
You are a business analyst assistant.

You have the following SALES and INVOICE context (plain text, already computed from the database):

{context_str[:8000]}

User question:
{user_query}

Task:
- Answer ONLY using the context above (do NOT invent numbers that are not implied there).
- If the context already includes information about top products, customers, revenues, etc.,
  reuse those numbers.
- If something is missing, say clearly what is missing instead of guessing.
"""
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=800,
            )
            reply = (resp.choices[0].message.content or "").strip()
            mem.last_user_query = user_query
            save_memory(db, mem)
            return OrchestratorResult(
                reply=reply or "I used your dashboard context, but it doesn’t include enough detail to answer that exactly.",
                action="new",
                reason=reason or "fallback_to_context_only",
                sql="",
                rows_preview=None,
                memory_updated=True,
            )
        # If we have neither a useful SQL result nor context, return a clear error.
        return OrchestratorResult(
            reply="I couldn’t generate a SQL query or find enough dashboard context to answer that. Try rephrasing with more detail (customer, product, country, and time period).",
            action="new",
            reason=reason or "sap_sql_agent_no_result",
        )

    # Summarize rows with LLM (same as sap_sql_agent summarizer, but we add memory/knowledge/context)
    preview = _rows_preview(result.rows, limit=20)
    prompt = f"""
You are an expert SAP sales/finance analyst.

Dashboard context (optional):
{context_str[:6000]}

Saved knowledge/notes (optional):
{json.dumps(mem.knowledge(), indent=2, default=str)[:4000]}

User question:
{user_query}

SQL executed:
```sql
{result.sql}
```

Result preview as JSON:
{json.dumps(preview, default=str)}

Task:
- Answer the question in clear business language.
- Include specific numbers when helpful.
- Be concise (3–10 sentences).
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=800,
    )
    reply = (resp.choices[0].message.content or "").strip()

    # Generate charts for visualization
    charts_data = None
    try:
        logger.info(f"Attempting chart generation for query with {len(result.rows)} rows")
        chart_specs = analyze_visualization_needs(result.rows, user_query, "new", result.sql)
        if chart_specs:
            charts_data = chart_specs_to_json(chart_specs)
            logger.info(f"✅ Generated {len(charts_data)} chart(s) for user query")
            logger.info(f"Chart types: {[c.get('chart_type') for c in charts_data]}")
        else:
            logger.info("No charts generated - analyze_visualization_needs returned empty list")
    except Exception as chart_err:
        logger.error(f"❌ Chart generation failed: {chart_err}", exc_info=True)

    mem.last_user_query = user_query
    mem.last_sql = result.sql
    mem.last_rows_json = json.dumps(_rows_preview(result.rows, limit=80), default=str)
    save_memory(db, mem)

    # Log to training data for fine-tuning
    try:
        log_query_execution(
            db=db,
            user_id=user_id,
            user_query=user_query,
            sql_query=result.sql,
            result_summary=reply,
            action_type="new",
            metadata={
                "rows_count": len(result.rows),
                "has_charts": bool(charts_data),
                "chart_count": len(charts_data) if charts_data else 0,
            },
        )
    except Exception as log_err:
        logger.warning(f"Failed to log training data: {log_err}")

    # Cache the result for future queries
    try:
        cache_query_result(
            db=db,
            query_text=user_query,
            sql_query=result.sql,
            result_summary=reply,
            result_preview=preview,
            charts=charts_data,
            ttl_hours=24,
        )
    except Exception as cache_err:
        logger.warning(f"Failed to cache query result: {cache_err}")

    return OrchestratorResult(
        reply=reply or "Query executed, but I couldn’t generate a summary.",
        action="new",
        reason=reason,
        sql=result.sql,
        rows_preview=preview,
        memory_updated=True,
        charts=charts_data,
    )


def orchestrator_payload(result: OrchestratorResult) -> Dict[str, Any]:
    payload = asdict(result)
    # keep payload small and frontend-safe
    if payload.get("rows_preview") is not None and len(payload["rows_preview"]) > 30:
        payload["rows_preview"] = payload["rows_preview"][:30]
    return payload

