# AI Training Logic (User Instructions)

The Zodiac AI can remember user instructions and apply them in future queries. No fine-tuning or model retraining is involved—instructions are stored as text and injected into prompts.

## Where It Lives

| File | Role |
|------|------|
| `app/services/ai_analysis_orchestrator.py` | Detects save instructions, decides action, calls `upsert_knowledge()` |
| `app/services/ai_analysis_memory_store.py` | Stores/loads user knowledge in `ai_analysis_memory.knowledge_json` |
| `app/services/sap_sql_agent.py` | Receives `knowledge_context` and uses it in prompts |

## How It Works

1. **Detection** – The orchestrator checks each user message:
   - **Explicit**: "Remember:", "Save this:", "Note:", "For future:", etc.
   - **Implicit**: "always add period", "use EKPO for cost", "include quantity", etc.

2. **Storage** – When detected, `upsert_knowledge()` adds the instruction to `knowledge_json` (per user, in `ai_analysis_memory`).

3. **Usage** – On every SQL-related query, the orchestrator passes `knowledge_context` (all saved notes) into `run_sap_sql_agent()`. The agent injects this into the LLM prompt so the model follows those rules when generating SQL.

## Few-Shot Training (Query→SQL Examples)

Successful query executions are logged to `ai_training_data`. The agent uses up to 3 past examples as few-shot prompts when generating SQL:

1. **Storage**: `log_query_execution()` saves (user_query, sql_query, result_summary) after each successful run.
2. **Retrieval**: `get_few_shot_examples(db, limit=3, user_id=...)` fetches recent examples, preferring high feedback_score.
3. **Usage**: Examples are injected into the `_generate_sql_json` prompt so the model learns from past patterns.

The more queries users run (and optionally rate via feedback), the better the model performs on similar questions.

## Extending the Logic

### Add more explicit prefixes

Edit `_is_explicit_knowledge_instruction()` in `ai_analysis_orchestrator.py`:

```python
prefixes = (
    "remember:",
    "save this:",
    # Add: "from now on:", "prefer:", etc.
)
```

### Add more implicit patterns

Edit `_is_implicit_knowledge_instruction()` in `ai_analysis_orchestrator.py`:

```python
patterns = (
    "always add",
    "always include",
    "add period",
    # Add: "prefer table", "use column", etc.
)
```

### Change storage format

Edit `ai_analysis_memory_store.py` – `knowledge_json` is a dict: `{key: value}`. You can add structure (e.g. categories, expiry) by changing `upsert_knowledge()` and how `knowledge_context` is built in the orchestrator.

### Use knowledge in prompts

The `knowledge_context` string is passed to `run_sap_sql_agent()`. Check `sap_sql_agent.py` for where it's injected into the prompt; you can adjust wording or position for better adherence.
