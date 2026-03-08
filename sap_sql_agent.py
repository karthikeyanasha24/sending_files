"""
Dynamic NL-to-SQL agent for SAP-style tables stored in the main Postgres DB.

This mirrors the INVOICE_BOT behaviour at a high level:
- LLM picks relevant tables based on the question and table descriptions.
- LLM produces a JSON SQL specification (tables, columns, joins, filters, order_by, limit).
- We convert that JSON spec into real SQL for Postgres and execute it via SQLAlchemy.
- Results are summarized back to the user via another LLM call.

It is intentionally generic and works over the following tables (if present in the DB):
VBRP, VBRK, VBAK, VBAP, VBEP, BSAD, BSEG, FAGLFLEXA, KNA1, KNVP, KNVV, MAKT, MARC, MARM, MEAN, MVKE.

The goal is to power questions like:
- "show me highest sales by product"
- "compare sales data with invoice data and identify major differences"
- "show me lowest sales by customer and country"
- "show me sales by country and industry"
- "which industry has highest revenues"
- "show me highest sales by customer and product and country"
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from ..config.config import OPENAI_API_KEY

logger = logging.getLogger("zodiac-api.sap_sql_agent")

try:
    from openai import OpenAI

    _openai_available = True
except ImportError:  # pragma: no cover - runtime dependency
    OpenAI = None  # type: ignore
    _openai_available = False


# --- Static metadata ---------------------------------------------------------------------------

SAP_TABLE_DESCRIPTIONS: Dict[str, str] = {
    # Sales / Billing
    "VBRP": "Billing document item (sales by product, quantities, net values, currencies, customers). Use for: highest sales by product, revenue analysis.",
    "vbrp": "Same as VBRP – billing document item. Use for sales, revenue, product analysis.",
    "VBRK": "Billing document header (invoice-level amounts, dates, currencies, customers).",
    "VBAK": "Sales document header (orders, customers, dates, overall values).",
    "VBAP": "Sales document item (ordered products, quantities, values).",
    "VBEP": "Schedule lines for sales document items (delivery quantities and dates).",
    # Customer / Material Master
    "KNA1": "Customer master (names, addresses, countries, industries).",
    "KNVV": "Customer sales data (sales area, pricing, related attributes).",
    "KNVP": "Customer partners (payer, ship-to, bill-to relationships).",
    "MAKT": "Material descriptions (product names).",
    "MARC": "Plant data for material (plant-level material attributes).",
    "MARM": "Units of measure for material (UOM conversion).",
    "MEAN": "International Article Numbers (EAN/UPC) for materials.",
    "MVKE": "Sales data for materials (sales org, distribution channel, pricing group).",
    # Finance / Accounting
    "BSAD": "Customer open and cleared items (AR line items, payments).",
    "BSEG": "Accounting document segment (line items for GL, customers, vendors).",
    "FAGLFLEXA": "General ledger: totals/line items for new G/L accounting.",
    # Logistics – Outbound
    "LIKP": "Outbound delivery header (delivery documents, shipping dates, quantities, ship-to). Use for: deliveries, logistics.",
    "LIPS": "Outbound delivery item (products, quantities, reference to sales order). Use for: delivery line details.",
    # Vendor / Purchasing
    "LFA1": "Vendor master (vendor names, addresses, countries). Use for: supplier analysis, purchasing.",
    "LFB1": "Vendor company code (vendor accounting, payment terms).",
    "LFM1": "Vendor purchasing org (vendor–purchasing org data).",
    "EKKO": "Purchasing document header (PO header, vendor, dates, currency). Use for: purchase orders.",
    "EKPO": "Purchasing document item (PO line items, materials, quantities, values).",
    "EBAN": "Purchase requisition (requisition items, materials, quantities).",
    # Invoice Verification (Vendor Invoices)
    "RBKP": "Vendor invoice header (invoice document, vendor, amount, currency). Use for: vendor invoice analysis.",
    "RSEG": "Vendor invoice item (invoice line items, materials, quantities, amounts).",
    # Material Document / Reservations
    "MKPF": "Material document header (goods movement header, posting date).",
    "RESB": "Reservation/dependent requirements (material reservations, requirements).",
    "LSEG": "Document segment (document item data).",
}


# High-level join hints between common SAP tables. This is injected into the LLM
# prompt so that the SQL JSON spec uses realistic join paths instead of guessing.
SAP_JOIN_HINTS = """
Typical business key joins you MUST prefer (do NOT invent other join columns):

SALES / BILLING:
- VBRP (billing items) <-> VBRK (billing header)
  * VBRP.VBELN = VBRK.VBELN
- vbrp: same as VBRP, use vbrp.VBELN = VBRK.VBELN

- VBRK (billing header) <-> KNA1 (customer master)
  * VBRK.KUNAG = KNA1.KUNNR
  * VBRK.KUNRG = KNA1.KUNNR   (payer, if present)

- VBAK (sales order header) <-> VBAP (sales order items)
  * VBAK.VBELN = VBAP.VBELN

- VBAP (order items) / VBRP (billing items) <-> VBEP (schedule lines)
  * VBEP.VBELN = VBAP.VBELN AND VBEP.POSNR = VBAP.POSNR

- VBRP / VBAP / LIPS (items with product) <-> MAKT / MVKE / MARC (material master & descriptions)
  * VBRP.MATNR = MAKT.MATNR = MVKE.MATNR = MARC.MATNR
  * VBAP.MATNR = MAKT.MATNR
  * LIPS.MATNR = MAKT.MATNR

- BSAD / BSEG (AR items) <-> KNA1 (customer master)
  * BSAD.KUNNR = KNA1.KUNNR
  * BSEG.KUNNR = KNA1.KUNNR

LOGISTICS – OUTBOUND DELIVERY:
- LIKP (delivery header) <-> LIPS (delivery items)
  * LIKP.VBELN = LIPS.VBELN

- LIPS (delivery items) <-> VBAP (sales order items) – reference
  * LIPS.VGBEL = VBAP.VBELN AND LIPS.VGPOS = VBAP.POSNR

- LIKP (delivery) <-> KNA1 (ship-to customer)
  * LIKP.KUNNR = KNA1.KUNNR  (if present)

PURCHASING / VENDOR:
- EKKO (PO header) <-> EKPO (PO items)
  * EKKO.EBELN = EKPO.EBELN

- EKKO / EKPO <-> LFA1 (vendor master)
  * EKKO.LIFNR = LFA1.LIFNR
  * EKPO.LIFNR = LFA1.LIFNR  (if present)

- RBKP (vendor invoice header) <-> RSEG (vendor invoice items)
  * RBKP.BELNR = RSEG.BELNR AND RBKP.GJAHR = RSEG.GJAHR  (if GJAHR present)
  * or RBKP.BELNR = RSEG.BELNR

- RBKP <-> LFA1 (vendor)
  * RBKP.LIFNR = LFA1.LIFNR

- EBAN (purchase req) <-> EKPO (PO items) – optional, via EBAN–EKPO reference fields if present

MATERIAL DOCUMENT:
- MKPF <-> MSEG (if MSEG exists): MKPF.MBLNR = MSEG.MBLNR, MKPF.MJAHR = MSEG.MJAHR

VERY IMPORTANT:
- VBRP / vbrp usually does NOT have KUNNR directly. To reach the customer, go:
  VBRP.VBELN -> VBRK.VBELN, then VBRK.KUNAG -> KNA1.KUNNR.
- When you need INDUSTRY or COUNTRY of a customer, read from:
  * KNA1.BRSCH (industry)
  * KNA1.LAND1 (country)
- For cost-related or COGS queries, use EKPO (purchase values), RBKP/RSEG (vendor invoice amounts), BSEG (accounting).
"""


@dataclass
class SqlAgentResult:
    sql: str
    rows: List[Dict[str, Any]]


_QUERY_TO_SQL_CACHE: Dict[str, str] = {}
_SQL_TO_ROWS_CACHE: Dict[str, List[Dict[str, Any]]] = {}


# --- Utility helpers --------------------------------------------------------------------------


def _get_openai_client() -> OpenAI | None:
    if not _openai_available:
        logger.warning("OpenAI package not available for sap_sql_agent")
        return None
    api_key = OPENAI_API_KEY
    if not api_key:
        logger.warning("OPEN_AI_KEY/OPENAI_API_KEY not set for sap_sql_agent")
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:  # pragma: no cover - network/config
        logger.warning("Failed to create OpenAI client for sap_sql_agent: %s", e)
        return None


def _serialize_value(v: Any) -> Any:
    if isinstance(v, Decimal):
        return float(v)
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="ignore")
    return v


def _load_table_mapping_file(root: Path, table_name: str) -> Dict[str, str]:
    """Load app/table_mapping/{TABLE}.json (invoice-bot format: {"COLUMN": "human_name"})."""
    import json as _json
    for candidate in (table_name, table_name.upper(), table_name.lower()):
        path = root / "table_mapping" / f"{candidate}.json"
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    raw = _json.load(f)
                if isinstance(raw, dict):
                    return {str(k): str(v) for k, v in raw.items()}
            except Exception:
                pass
    return {}


def _introspect_columns(db: Session, table_names: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Introspect the DB to get columns for each table and build semantic descriptions.

    Priority:
    1) db_table_mapping.json (generated by test_db_connection.py) per-column descriptions
    2) app/table_mapping/{TABLE}.json (invoice-bot format: {"COLUMN": "human_name"})
    3) Simple heuristic descriptions
    """
    insp = inspect(db.bind)
    mapping: Dict[str, Dict[str, str]] = {}
    root = Path(__file__).resolve().parent.parent

    # Try to load pre-generated mapping file (optional)
    mapping_file: Dict[str, Any] = {}
    try:
        path = root / "db_table_mapping.json"
        if path.exists():
            import json as _json

            with path.open("r", encoding="utf-8") as f:
                raw = _json.load(f)
            if isinstance(raw, dict):
                mapping_file = raw
    except Exception:
        mapping_file = {}

    for tbl in table_names:
        # Find matching table in DB, case-insensitive
        db_tables = {t.lower(): t for t in insp.get_table_names()}
        actual_name = db_tables.get(tbl.lower())
        if not actual_name:
            continue

        cols: Dict[str, str] = {}
        file_entry = mapping_file.get(actual_name) or mapping_file.get(tbl) or {}
        file_cols = (file_entry.get("columns") or {}) if isinstance(file_entry, dict) else {}

        # Load invoice-bot table_mapping/{TABLE}.json when db_table_mapping lacks column
        table_mapping_cols = _load_table_mapping_file(root, actual_name)

        for col in insp.get_columns(actual_name):
            col_name = col["name"]
            if isinstance(file_cols, dict) and col_name in file_cols:
                cols[col_name] = str(file_cols[col_name])
            elif table_mapping_cols and col_name in table_mapping_cols:
                cols[col_name] = str(table_mapping_cols[col_name])
            else:
                cols[col_name] = f"{tbl}.{col_name} column"

        if cols:
            mapping[actual_name] = cols

    return mapping


def _get_table_descriptions(db: Session) -> Dict[str, str]:
    """
    Build table descriptions by merging:
    1) SAP_TABLE_DESCRIPTIONS (known tables with semantic descriptions)
    2) db_table_mapping.json meta (for tables in DB, when mapping exists)
    3) Generic fallback for any other DB table

    This allows new tables added to the DB + mapping to be automatically
    discoverable by the model without code changes.
    """
    insp = inspect(db.bind)
    db_table_names = insp.get_table_names()
    # Skip non-SAP / internal tables for AI analysis
    skip = {"ai_analysis_memory", "zodiac_users", "zodiac_customers", "customers", "user_customers"}
    db_table_names = [t for t in db_table_names if t.lower() not in skip]

    mapping_file: Dict[str, Any] = {}
    try:
        root = Path(__file__).resolve().parent.parent
        path = root / "db_table_mapping.json"
        if path.exists():
            import json as _json
            with path.open("r", encoding="utf-8") as f:
                raw = _json.load(f)
            if isinstance(raw, dict):
                mapping_file = raw
    except Exception:
        pass

    out: Dict[str, str] = {}
    for actual_name in db_table_names:
        # Prefer SAP_TABLE_DESCRIPTIONS (supports both VBRP and vbrp)
        desc = SAP_TABLE_DESCRIPTIONS.get(actual_name) or SAP_TABLE_DESCRIPTIONS.get(actual_name.upper())
        if desc:
            out[actual_name] = desc
            continue
        # Else use mapping meta
        entry = mapping_file.get(actual_name) or mapping_file.get(actual_name.upper())
        if isinstance(entry, dict) and entry.get("meta", {}).get("description"):
            out[actual_name] = str(entry["meta"]["description"])
        else:
            out[actual_name] = f"{actual_name} table – check columns for available fields."
    return out


def _pick_tables(
    question: str,
    client: OpenAI,
    db: Session,
    knowledge_context: Optional[str] = None,
) -> List[str]:
    """
    Pick tables needed to answer the question. Uses dynamic table list from DB +
    mapping so new tables are auto-detected. User knowledge (e.g. "use these for costs")
    is injected when provided.
    """
    table_descriptions = _get_table_descriptions(db)
    if not table_descriptions:
        table_descriptions = SAP_TABLE_DESCRIPTIONS  # fallback

    knowledge_block = ""
    if knowledge_context and knowledge_context.strip():
        knowledge_block = f"""
User preferences / stored knowledge (apply when relevant):
{knowledge_context.strip()}
"""

    prompt = f"""
User question: "{question}"
{knowledge_block}
You are selecting SAP-style tables that live in a Postgres database.
Here are the available tables (use exact names as shown):
{json.dumps(table_descriptions, indent=2)}

Task:
- Choose ONLY the tables that are truly needed to answer the question.
- Use EXACT table names as they appear above (e.g. vbrp if listed, VBRK, LIKP, etc.).
- For sales/revenue: prefer VBRP or vbrp + VBRK + KNA1 + MAKT.
- For purchasing/vendor invoices: prefer EKKO + EKPO + LFA1, or RBKP + RSEG + LFA1.
- For logistics/deliveries: prefer LIKP + LIPS.
- Return STRICT JSON only:
{{
  "selected_tables": [
    {{ "name": "<exact_table_name>", "description": "..." }}
  ]
}}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = resp.choices[0].message.content or ""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
    tables = [t.get("name") for t in data.get("selected_tables", []) if isinstance(t, dict) and t.get("name")]
    # Normalize: ensure we only pick tables that exist in DB
    db_tables_lower = {t.lower(): t for t in table_descriptions}
    normalized = []
    for t in tables:
        key = (t or "").strip()
        if not key:
            continue
        actual = db_tables_lower.get(key.lower())
        if actual:
            normalized.append(actual)
    tables = normalized
    if not tables:
        # Fallback: try vbrp/VBRP, VBRK, or first available
        for cand in ["vbrp", "VBRP", "VBRK"]:
            if cand in table_descriptions or cand.upper() in {k.upper() for k in table_descriptions}:
                tables = [cand if cand in table_descriptions else next(k for k in table_descriptions if k.upper() == cand.upper())]
                break
        if not tables and table_descriptions:
            tables = [list(table_descriptions.keys())[0]]

    # Force-add MAKT when product/sales data is used so product names (MAKTX) are shown, not only material numbers
    product_sales_keywords = ("product", "sales", "revenue", "sold", "highest", "top", "material")
    q_lower = (question or "").lower()
    has_product_question = any(k in q_lower for k in product_sales_keywords)
    product_line_upper = {"VBRP", "VBAP", "LIPS"}
    has_product_line = any((t or "").upper() in product_line_upper for t in tables)
    if has_product_question and has_product_line:
        makt_candidate = next((k for k in table_descriptions if k.upper() == "MAKT"), None)
        if makt_candidate and makt_candidate not in tables:
            tables = list(tables) + [makt_candidate]

    return tables


def _generate_sql_json(
    question: str,
    selected_tables: List[str],
    column_mappings: Dict[str, Dict[str, str]],
    client: OpenAI,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Equivalent of INVOICE_BOT.generate_sql_json, but simplified and Postgres-focused.

    Supports:
    - simple SELECTs
    - aggregates (SUM/AVG/COUNT/MIN/MAX) via the optional "agg" field on columns
    - GROUP BY via a dedicated "group_by" list
    - few_shot_examples: past query→SQL pairs to train by example
    """
    few_shot_block = ""
    if few_shot_examples:
        examples_text = "\n\n".join(
            f'Question: "{ex.get("user_query", "")}"\nSQL: {ex.get("sql_query", "")}'
            for ex in few_shot_examples[:3]
        )
        few_shot_block = f"""
Examples from past successful queries (learn from these patterns):
{examples_text}

"""
    prompt = f"""
{few_shot_block}User question: "{question}"

Tables available (subset already selected as relevant):
{json.dumps({tbl: SAP_TABLE_DESCRIPTIONS.get(tbl, "") for tbl in selected_tables}, indent=2)}

Column mappings (table -> column -> short description):
{json.dumps(column_mappings, indent=2)}

Known join patterns between these tables:
{SAP_JOIN_HINTS}

Task:
- Choose relevant columns from these tables.
- Propose joins between tables using ONLY the business keys listed above (do NOT invent other join columns).
- Remember that VBRP typically does NOT have KUNNR; to reach the customer, you MUST join via VBRK then KNA1.
- Add filters only if clearly needed from the question (for dates, customers, countries, industries, products, etc.).
- Return STRICT JSON with this structure:
{{
  "tables": [{{ "name": "VBRP", "description": "..." }}],
  "columns": [
    {{ "table": "KNA1", "name": "BRSCH", "description": "industry of the customer", "agg": null }},
    {{ "table": "VBRP", "name": "NETWR", "description": "billing item net value", "agg": "SUM" }}
  ],
  "joins": [{{ "left": "VBRP", "right": "VBRK", "on": "VBRP.VBELN = VBRK.VBELN" }}],
  "filters": [{{ "lhs": "VBRK.FKDAT", "operator": ">=", "rhs": "'2024-01-01'" }}],
  "group_by": [
    {{ "table": "KNA1", "column": "BRSCH" }}
  ],
  "order_by": [
    "total_sales DESC"
  ],
  "limit": 200
}}

Rules:
- Use table and column names that actually exist in the column mappings.
- If a column entry has "agg": "SUM" | "AVG" | "COUNT" | "MIN" | "MAX",
  you are defining an aggregated metric over that column.
- For SALES / REVENUE / PRODUCT questions (e.g. "highest sales by product", "top products"):
  * ALWAYS include MAKT.MAKTX (product name / material description) so results show product names, NOT only material numbers.
  * When VBRP or VBRK is used: include VBRK.WAERK or VBRP.WAERK (currency) and VBRP.FKIMG (billed quantity).
  * Include VBRK.FKDAT (billing date) or GJAHR/POPER for period/year when the question asks for year, period, or monthly breakdown.
  * Join MAKT on MAKT.MATNR = VBRP.MATNR (or VBAP.MATNR, LIPS.MATNR); use MAKT.SPRAS = 'E' for English when filtering.
- For questions like:
    * "which industry has highest revenues"
    * "top customers by sales"
    * "highest sales by customer and product"
  you MUST:
    * pick an appropriate numeric metric column (e.g., VBRP.NETWR, BSAD.DMBTR, INVOICE_V2_BUSINESS_DATA.TOTAL_AMOUNT)
    * set "agg": "SUM" (or another relevant aggregate) on that metric column
    * add the dimension columns (industry, customer, product, country, etc.) to "group_by"
    * filter out NULL dimension values where it makes sense (e.g., industry IS NOT NULL)
    * order by the aggregated metric (e.g., "total_sales DESC") and use a small limit (e.g. 50 or 100).
- If the question is about "lowest", sort ASC instead of DESC.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = resp.choices[0].message.content or ""
    try:
        spec = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        spec = json.loads(m.group(0)) if m else {}
    return spec or {}


def _ensure_product_context_in_spec(
    spec: Dict[str, Any],
    question: str,
    selected_tables: List[str],
    column_mappings: Dict[str, Dict[str, str]],
    db: Session,
) -> None:
    """
    Post-process spec to ensure product names, currency, quantity, and period
    are included for product/sales queries (client requirement).
    """
    if not spec:
        return
    q_lower = (question or "").lower()
    product_sales_keywords = ("product", "sales", "revenue", "sold", "highest", "top", "material", "value")
    year_period_keywords = ("year", "period", "month", "monthly", "date")
    is_product_sales = any(k in q_lower for k in product_sales_keywords)
    wants_year_period = any(k in q_lower for k in year_period_keywords)

    tables_info = spec.get("tables") or []
    columns = spec.get("columns") or []
    joins = spec.get("joins") or []

    table_names = set()
    for t in tables_info:
        if isinstance(t, dict) and t.get("name"):
            table_names.add(t["name"])
    for c in columns:
        if isinstance(c, dict) and c.get("table"):
            table_names.add(c["table"])
    for j in joins:
        if isinstance(j, dict):
            if j.get("left"):
                table_names.add(j["left"])
            if j.get("right"):
                table_names.add(j["right"])

    col_tuples = {(c.get("table"), c.get("name")) for c in columns if isinstance(c, dict) and c.get("table") and c.get("name")}

    def has_column(tbl: str, col: str) -> bool:
        return (tbl, col) in col_tuples

    def table_in_spec(tbl: str) -> bool:
        return any((t or "").upper() == tbl.upper() for t in table_names)

    changed = False
    product_line_tables = {"VBRP", "vbrp", "VBAP", "LIPS"}
    spec_has_product_line = any(t.upper() in {p.upper() for p in product_line_tables} for t in table_names)

    if is_product_sales and spec_has_product_line:
        makt_name = next((t for t in table_names if (t or "").upper() == "MAKT"), None)
        if not makt_name:
            makt_intro = _introspect_columns(db, ["MAKT"])
            if makt_intro:
                makt_name = list(makt_intro.keys())[0]
                column_mappings.update(makt_intro)
                if makt_name not in selected_tables:
                    selected_tables.append(makt_name)
                tables_info.append({"name": makt_name, "description": "Material descriptions (product names)"})
                item_table = next((t for t in table_names if (t or "").upper() in {"VBRP", "VBAP", "LIPS"}), None)
                if item_table:
                    joins.append({
                        "left": item_table,
                        "right": makt_name,
                        "on": f"{item_table}.MATNR = {makt_name}.MATNR",
                    })
                columns.append({"table": makt_name, "name": "MAKTX", "description": "product name", "agg": None})
                col_tuples.add((makt_name, "MAKTX"))
                table_names.add(makt_name)
                changed = True
        if makt_name and not has_column(makt_name, "MAKTX"):
            makt_cols = column_mappings.get(makt_name, {}) or {}
            col_name = next((c for c in makt_cols if (c or "").upper() == "MAKTX"), "MAKTX")
            columns.append({"table": makt_name, "name": col_name, "description": "product name", "agg": None})
            col_tuples.add((makt_name, col_name))
            changed = True

        vbrk_name = next((t for t in table_names if (t or "").upper() == "VBRK"), None)
        vbrp_name = next((t for t in table_names if (t or "").upper() == "VBRP"), None)
        if (vbrk_name or vbrp_name) and not any(has_column(t, "WAERK") for t in table_names if t):
            tbl = vbrk_name or vbrp_name
            tbl_cols = column_mappings.get(tbl, {})
            waerk_col = next((c for c in tbl_cols if c.upper() == "WAERK"), None)
            if waerk_col or "WAERK" in (tbl_cols or {}):
                col = waerk_col or "WAERK"
                columns.append({"table": tbl, "name": col, "description": "currency", "agg": None})
                changed = True
        if vbrp_name and not has_column(vbrp_name, "FKIMG"):
            vbrp_cols = column_mappings.get(vbrp_name, {})
            fkimg_col = next((c for c in vbrp_cols if c.upper() == "FKIMG"), None)
            if fkimg_col or "FKIMG" in (vbrp_cols or {}):
                col = fkimg_col or "FKIMG"
                columns.append({"table": vbrp_name, "name": col, "description": "billed quantity", "agg": "SUM"})
                changed = True
        if wants_year_period and vbrk_name and not has_column(vbrk_name, "FKDAT"):
            vbrk_cols = column_mappings.get(vbrk_name, {})
            fkdat_col = next((c for c in vbrk_cols if c.upper() == "FKDAT"), None)
            if fkdat_col or "FKDAT" in (vbrk_cols or {}):
                col = fkdat_col or "FKDAT"
                columns.append({"table": vbrk_name, "name": col, "description": "billing date", "agg": None})
                changed = True

    if changed:
        spec["tables"] = tables_info
        spec["columns"] = columns
        spec["joins"] = joins


def _json_to_sql_postgres(json_spec: Dict[str, Any], column_mappings: Dict[str, Dict[str, str]]) -> str:
    """
    Convert JSON spec into a Postgres SQL query.

    We mirror the INVOICE_BOT approach:
    - Build SELECT list for chosen columns.
    - Start from a base table, then add JOINs from the spec.
    - If some tables are missing join conditions, join them via common keys if possible (e.g., VBELN, KUNNR, MATNR).
    """
    columns = json_spec.get("columns", []) or []
    joins = json_spec.get("joins", []) or []
    tables_info = json_spec.get("tables", []) or []
    limit = int(json_spec.get("limit", 100) or 100)

    all_tables: List[str] = []
    for j in joins:
        all_tables.append(j.get("left"))
        all_tables.append(j.get("right"))
    for c in columns:
        all_tables.append(c.get("table"))
    for t in tables_info:
        all_tables.append(t.get("name"))
    all_tables = [t for t in {t for t in all_tables if t}]  # unique, remove None

    if not all_tables:
        raise ValueError("sap_sql_agent: JSON spec contains no tables")

    # Map spec table names to actual DB table names (case differences already handled by column_mappings keys)
    table_aliases: Dict[str, str] = {}
    used_aliases: set[str] = set()

    def _fmt_alias(tbl: str) -> str:
        base = tbl[0].lower()
        alias = base
        i = 1
        while alias in used_aliases:
            alias = f"{base}{i}"
            i += 1
        used_aliases.add(alias)
        return alias

    # The mapping keys in column_mappings are the actual DB table names.
    # Build a map from upper-case logical name -> actual DB table name.
    logical_to_actual: Dict[str, str] = {}
    for logical in SAP_TABLE_DESCRIPTIONS.keys():
        for actual in column_mappings.keys():
            if actual.lower() == logical.lower():
                logical_to_actual[logical] = actual

    def _actual_table_name(logical: str) -> str:
        return logical_to_actual.get(logical, logical)

    # Build a case-insensitive column name map per table so we can always
    # use the real DB column identifiers even if the JSON spec uses upper-case.
    column_name_map: Dict[str, Dict[str, str]] = {}
    for actual_tbl, cols in column_mappings.items():
        column_name_map[actual_tbl] = {c_name.lower(): c_name for c_name in cols.keys()}

    def _actual_column_name(actual_tbl: str, logical_col: str) -> str:
        if not logical_col:
            return logical_col
        table_cols = column_name_map.get(actual_tbl, {})
        return table_cols.get(str(logical_col).lower(), logical_col)

    for tbl in all_tables:
        actual = _actual_table_name(tbl)
        table_aliases[actual] = _fmt_alias(actual)

    # SELECT
    select_parts: List[str] = []
    used_col_aliases: set[str] = set()

    for col in columns:
        logical_tbl = col.get("table")
        col_name_raw = col.get("name")
        if not logical_tbl or not col_name_raw:
            continue
        actual_tbl = _actual_table_name(logical_tbl)
        if actual_tbl not in table_aliases:
            continue
        alias = table_aliases[actual_tbl]
        col_name = _actual_column_name(actual_tbl, col_name_raw)
        agg = str(col.get("agg") or "").upper()
        if agg in {"SUM", "AVG", "COUNT", "MIN", "MAX"}:
            expr = f"{agg}({alias}.\"{col_name}\")"
        else:
            expr = f'{alias}."{col_name}"'
        human = col.get("description") or f"{actual_tbl}_{col_name}"
        human_safe = re.sub(r"[^\w]", "_", human)[:60] or f"{alias}_{col_name}"
        if human_safe in used_col_aliases:
            suffix = 1
            while f"{human_safe}_{suffix}" in used_col_aliases:
                suffix += 1
            human_safe = f"{human_safe}_{suffix}"
        used_col_aliases.add(human_safe)
        select_parts.append(f'    {expr} AS "{human_safe}"')

    if not select_parts:
        # Fallback: SELECT * from first table
        base_logical = all_tables[0]
        base_actual = _actual_table_name(base_logical)
        alias = table_aliases[base_actual]
        select_parts.append(f"    {alias}.*")

    sql_lines: List[str] = [f"SELECT {', '.join(select_parts)}",]

    # Base table: first in joins, else first in tables_info, else first in list
    base_logical = (
        (joins[0].get("left") if joins else None)
        or (tables_info[0].get("name") if tables_info else None)
        or all_tables[0]
    )
    base_actual = _actual_table_name(base_logical)
    base_alias = table_aliases[base_actual]
    sql_lines.append(f'FROM "{base_actual}" AS {base_alias}')

    added_actuals = {base_actual}

    # Helper to rewrite "VBRP.VBELN" → "p.\"VBELN\"" using aliases and actual names
    def _rewrite_expr(expr: str) -> str:
        out = expr
        for logical, actual in logical_to_actual.items():
            alias = table_aliases.get(actual)
            if not alias:
                continue
            out = re.sub(rf"\b{logical}\.", f"{alias}.", out)
        # If already uses actual names, also replace them
        for actual, alias in table_aliases.items():
            out = re.sub(rf"\b{actual}\.", f"{alias}.", out)
        return out

    # Add joins from spec
    for j in joins:
        left_logical = j.get("left")
        right_logical = j.get("right")
        on_expr = j.get("on") or ""
        if not right_logical:
            continue
        right_actual = _actual_table_name(right_logical)
        if right_actual in added_actuals:
            continue
        right_alias = table_aliases[right_actual]
        sql_lines.append(f'\nLEFT JOIN "{right_actual}" AS {right_alias}')
        if on_expr:
            sql_lines.append(f"    ON {_rewrite_expr(on_expr)}")
        added_actuals.add(right_actual)

    # Add any missing tables with heuristic joins on common keys
    COMMON_KEYS = ["VBELN", "KUNNR", "KUNAG", "MATNR", "EBELN", "LIFNR", "BELNR"]
    for logical_tbl in all_tables:
        actual_tbl = _actual_table_name(logical_tbl)
        if actual_tbl in added_actuals:
            continue
        alias = table_aliases[actual_tbl]
        sql_lines.append(f'\nLEFT JOIN "{actual_tbl}" AS {alias}')
        # Try to join on a shared key with base table (case-insensitive column resolution)
        join_cond = None
        for key in COMMON_KEYS:
            base_col = _actual_column_name(base_actual, key)
            other_col = _actual_column_name(actual_tbl, key)
            if base_col and other_col:
                join_cond_candidate = f'{base_alias}."{base_col}" = {alias}."{other_col}"'
                join_cond = join_cond_candidate
                break
        if join_cond:
            sql_lines.append(f"    ON {join_cond}")
        added_actuals.add(actual_tbl)

    # GROUP BY
    group_bys = json_spec.get("group_by", []) or []
    gb_parts: List[str] = []
    for gb in group_bys:
        if not isinstance(gb, dict):
            continue
        t_logical = gb.get("table")
        col_raw = gb.get("column")
        if not t_logical or not col_raw:
            continue
        t_actual = _actual_table_name(t_logical)
        alias = table_aliases.get(t_actual)
        if not alias:
            continue
        col = _actual_column_name(t_actual, col_raw)
        gb_parts.append(f'{alias}."{col}"')
    if gb_parts:
        sql_lines.append("\nGROUP BY " + ", ".join(gb_parts))

    # WHERE
    conds: List[str] = []
    for f in json_spec.get("filters", []) or []:
        lhs = _rewrite_expr(str(f.get("lhs", "")))
        op = str(f.get("operator", "")).strip()
        rhs = str(f.get("rhs", "")).strip()
        if lhs and op and rhs:
            conds.append(f"{lhs} {op} {rhs}")
    if conds:
        sql_lines.append("\nWHERE " + " AND ".join(conds))

    # ORDER BY
    order_by_parts: List[str] = []
    for ob in json_spec.get("order_by", []) or []:
        if isinstance(ob, str):
            order_by_parts.append(_rewrite_expr(ob))
        else:
            t_logical = ob.get("table")
            col_raw = ob.get("column")
            direction = ob.get("direction", "DESC").upper()
            t_actual = _actual_table_name(t_logical) if t_logical else None
            alias = table_aliases.get(t_actual) if t_actual else None
            if alias and col_raw and t_actual:
                col = _actual_column_name(t_actual, col_raw)
                order_by_parts.append(f'{alias}."{col}" {direction}')
    if order_by_parts:
        sql_lines.append("\nORDER BY " + ", ".join(order_by_parts))

    sql_lines.append(f"\nLIMIT {limit}")
    return "\n".join(sql_lines) + ";"


def _run_sql(db: Session, sql: str) -> List[Dict[str, Any]]:
    if not sql or not sql.strip():
        return []
    try:
        result = db.execute(text(sql))
        rows = result.fetchall()
        keys = result.keys()
        out: List[Dict[str, Any]] = []
        for row in rows:
            row_dict = {k: _serialize_value(v) for k, v in zip(keys, row)}
            out.append(row_dict)
        return out
    except Exception as e:
        logger.warning("sap_sql_agent SQL execution failed: %s", e)
        return []


def _summarize_results(question: str, sql: str, rows: List[Dict[str, Any]], client: OpenAI) -> str:
    """
    Ask the LLM to summarize the tabular result in natural language.
    """
    if not rows:
        return ""

    # Truncate to keep prompt reasonable
    sample_rows = rows[:100]
    data_json = json.dumps(sample_rows, default=_serialize_value)

    prompt = f"""
You are an expert SAP sales and finance analyst.

The user asked:
\"\"\"{question}\"\"\"

You executed the following SQL on a Postgres database that contains SAP-style tables:
```sql
{sql}
```

Here is a sample of the result rows as JSON:
{data_json}

Task:
- Explain the answer to the user's question in clear business language.
- Include specific numbers (totals, top items, customers, countries, industries) when helpful.
- Be concise (3–8 sentences).
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=600,
    )
    return (resp.choices[0].message.content or "").strip()


def validate_sql_spec(spec: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate SQL specification before execution.
    
    Args:
        spec: JSON SQL specification
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check basic structure
    if not spec:
        errors.append("Empty specification")
        return False, errors
    
    tables = spec.get("tables", [])
    columns = spec.get("columns", [])
    
    if not tables:
        errors.append("No tables specified")
    
    if not columns:
        errors.append("No columns specified")
    
    # Validate columns reference existing tables
    table_names = {t.get("name") for t in tables if isinstance(t, dict) and t.get("name")}
    for col in columns:
        if isinstance(col, dict):
            col_table = col.get("table")
            if col_table and col_table not in table_names:
                errors.append(f"Column references unknown table: {col_table}")
    
    # Validate joins reference existing tables
    joins = spec.get("joins", [])
    for j in joins:
        if isinstance(j, dict):
            left = j.get("left")
            right = j.get("right")
            if left and left not in table_names:
                errors.append(f"Join references unknown left table: {left}")
            if right and right not in table_names:
                errors.append(f"Join references unknown right table: {right}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def refine_query_on_error(
    client: OpenAI,
    original_question: str,
    sql_error: str,
    previous_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Use LLM to refine the query specification after an SQL error.
    
    Args:
        client: OpenAI client
        original_question: User's original question
        sql_error: Error message from SQL execution
        previous_spec: Previous JSON SQL specification that failed
    
    Returns:
        Refined JSON SQL specification
    """
    try:
        prompt = f"""
An SQL query failed with an error. Please fix the JSON SQL specification.

Original question: "{original_question}"

Previous specification that failed:
{json.dumps(previous_spec, indent=2)}

Error message:
{sql_error}

Common issues:
- Missing join conditions
- Invalid column names
- Incorrect table references
- Missing GROUP BY for aggregated columns

Return a CORRECTED JSON specification with the same structure.
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000,
        )
        
        content = (response.choices[0].message.content or "").strip()
        try:
            refined_spec = json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            refined_spec = json.loads(m.group(0)) if m else previous_spec
        
        return refined_spec or previous_spec
    
    except Exception as e:
        logger.error(f"Query refinement failed: {e}")
        return previous_spec


def run_sap_sql_agent(
    question: str,
    db: Session,
    knowledge_context: Optional[str] = None,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
    max_retries: int = 2,
) -> SqlAgentResult | None:
    """
    Main entry point used by the dashboard AI endpoint.

    It:
    - Uses LLM to pick tables (dynamically from DB + mapping; supports new tables)
    - Introspects columns from Postgres
    - Uses LLM to build a JSON SQL spec (with optional few-shot examples for training)
    - Translates to Postgres SQL and executes
    - Caches SQL per question

    knowledge_context: Optional user preferences (e.g. "For cost queries use EKPO, RBKP, RSEG").
    few_shot_examples: Past query→SQL pairs from ai_training_data for few-shot prompting.
    """
    client = _get_openai_client()
    if not client:
        return None

    q_key = question.strip().lower()
    if q_key in _QUERY_TO_SQL_CACHE and _QUERY_TO_SQL_CACHE[q_key] in _SQL_TO_ROWS_CACHE:
        sql = _QUERY_TO_SQL_CACHE[q_key]
        rows = _SQL_TO_ROWS_CACHE[sql]
        return SqlAgentResult(sql=sql, rows=rows)

    attempt = 0
    last_error = None
    spec = None
    
    try:
        selected_tables = _pick_tables(question, client, db, knowledge_context)
        column_mappings = _introspect_columns(db, selected_tables)
        if not column_mappings:
            logger.warning("sap_sql_agent: no column mappings found for selected tables %s", selected_tables)
            return None

        spec = _generate_sql_json(
            question, selected_tables, column_mappings, client,
            few_shot_examples=few_shot_examples,
        )
        if not spec:
            logger.warning("sap_sql_agent: empty JSON spec for question %s", question)
            return None

        _ensure_product_context_in_spec(spec, question, selected_tables, column_mappings, db)

        # Validate specification
        is_valid, validation_errors = validate_sql_spec(spec)
        if not is_valid:
            logger.warning(f"Invalid SQL spec: {validation_errors}")
            # Try to auto-fix common issues
            if validation_errors and len(validation_errors) < 5:
                logger.info("Attempting to refine specification...")
                spec = refine_query_on_error(client, question, ", ".join(validation_errors), spec)
                is_valid, validation_errors = validate_sql_spec(spec)
        
        # Retry loop for SQL execution
        while attempt <= max_retries:
            try:
                sql = _json_to_sql_postgres(spec, column_mappings)
                rows = _run_sql(db, sql)
                
                # Success!
                if not rows:
                    logger.info("sap_sql_agent: SQL returned no rows for question %s", question)
                else:
                    _QUERY_TO_SQL_CACHE[q_key] = sql
                    _SQL_TO_ROWS_CACHE[sql] = rows
                
                return SqlAgentResult(sql=sql, rows=rows)
            
            except Exception as sql_err:
                last_error = str(sql_err)
                logger.warning(f"SQL execution failed (attempt {attempt + 1}/{max_retries + 1}): {last_error}")
                
                if attempt < max_retries:
                    # Try to refine the query
                    logger.info("Refining query specification...")
                    spec = refine_query_on_error(client, question, last_error, spec)
                    attempt += 1
                else:
                    # Max retries reached
                    logger.error(f"Max retries reached for question: {question}")
                    return None
        
        return None
    
    except Exception as e:
        logger.warning("sap_sql_agent failed for question '%s': %s", question, e)
        return None


def answer_with_sap_sql_agent(question: str, db: Session) -> str:
    """
    Convenience wrapper: run the SAP SQL agent and turn its result into a natural language answer.
    """
    client = _get_openai_client()
    if not client:
        return ""

    result = run_sap_sql_agent(question, db)
    if not result or not result.rows:
        return ""

    try:
        summary = _summarize_results(question, result.sql, result.rows, client)
    except Exception as e:
        logger.warning("sap_sql_agent summarization failed: %s", e)
        return ""

    return summary

