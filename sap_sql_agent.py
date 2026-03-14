"""
Dynamic NL-to-SQL agent for SAP-style tables stored in the main Postgres DB.

This mirrors the INVOICE_BOT behaviour at a high level:
- LLM picks relevant tables based on the question and table descriptions.
- LLM produces a JSON SQL specification (tables, columns, joins, filters, order_by, limit).
- We convert that JSON spec into real SQL for Postgres and execute it via SQLAlchemy.
- Results are summarized back to the user via another LLM call.

It is intentionally generic and works over the following tables (if present in the DB):
VBRP, VBRK, VBAK, VBAP, VBEP, BSAD, BSEG, FAGLFLEXA, KNA1, KNVP, KNVV, MAKT, MARC, MARM, MEAN, MVKE, T016T.

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

# Schema-driven agent (no keyword rules): schema → LLM table selection → LLM SQL
try:
    from .schema_loader import get_schema_dict, get_schema_text, schema_to_text
    from .semantic_sql_resolver import resolve_to_sql as semantic_resolve_to_sql
    from .semantic_sql_resolver import resolve_count_by_dimension as semantic_resolve_count_by
    from .table_selector_llm import select_tables as schema_select_tables
    from .sql_generator_llm import generate_sql as schema_generate_sql
    from .sql_validator import validate_sql as schema_validate_sql
    try:
        from .query_resolver import try_resolve_and_build_sql, get_semantic_context_for_prompt
    except ImportError:
        try_resolve_and_build_sql = None
        get_semantic_context_for_prompt = lambda: ""
    _SCHEMA_DRIVEN_AVAILABLE = True
except ImportError:
    _SCHEMA_DRIVEN_AVAILABLE = False
    try_resolve_and_build_sql = None

try:
    from openai import OpenAI

    _openai_available = True
except ImportError:  # pragma: no cover - runtime dependency
    OpenAI = None  # type: ignore
    _openai_available = False


# --- Schema config (extensible: edit schema_ai_config.json when adding tables) ---

def _load_schema_config() -> Dict[str, Any]:
    """Load schema_ai_config.json for extensible rules. Returns {} on failure."""
    try:
        root = Path(__file__).resolve().parent.parent
        path = root / "schema_ai_config.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg if isinstance(cfg, dict) else {}
    except Exception as e:
        logger.warning("Could not load schema_ai_config.json: %s", e)
    return {}


_SCHEMA_CONFIG: Dict[str, Any] = {}
def _get_schema_config() -> Dict[str, Any]:
    global _SCHEMA_CONFIG
    if not _SCHEMA_CONFIG:
        _SCHEMA_CONFIG = _load_schema_config()
    return _SCHEMA_CONFIG


def _get_semantic_prompt_block() -> str:
    """Return semantic dictionary context for SQL generator prompts. Empty if unavailable."""
    try:
        block = get_semantic_context_for_prompt()
        return f"\n{block}\n\n" if block else ""
    except Exception:
        return ""


# --- SQL Catalog: pre-built queries for common patterns ---

_SQL_CATALOG: Optional[List[Dict[str, Any]]] = None
_SQL_CATALOG_MTIME: float = 0.0  # last-modified time of the file when last loaded


def _load_sql_catalog() -> List[Dict[str, Any]]:
    """Load sql_catalog.json. Returns [] on failure."""
    try:
        root = Path(__file__).resolve().parent.parent
        path = root / "sql_catalog.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                catalog = json.load(f)
            if isinstance(catalog, list):
                logger.info("sql_catalog: loaded %d entries", len(catalog))
                return catalog
    except Exception as e:
        logger.warning("Could not load sql_catalog.json: %s", e)
    return []


def _get_sql_catalog() -> List[Dict[str, Any]]:
    """Return catalog, auto-reloading if sql_catalog.json has been modified on disk."""
    global _SQL_CATALOG, _SQL_CATALOG_MTIME
    try:
        root = Path(__file__).resolve().parent.parent
        path = root / "sql_catalog.json"
        current_mtime = path.stat().st_mtime if path.exists() else 0.0
        if _SQL_CATALOG is None or current_mtime != _SQL_CATALOG_MTIME:
            _SQL_CATALOG = _load_sql_catalog()
            _SQL_CATALOG_MTIME = current_mtime
            logger.info("sql_catalog: (re)loaded %d entries from disk", len(_SQL_CATALOG))
    except Exception as e:
        logger.warning("sql_catalog: mtime check failed (%s), using cached version", e)
        if _SQL_CATALOG is None:
            _SQL_CATALOG = _load_sql_catalog()
    return _SQL_CATALOG or []


def _get_actual_table_names_from_mapping() -> set:
    """Return set of actual table names from db_table_mapping.json (used to quote catalog SQL)."""
    try:
        root = Path(__file__).resolve().parent.parent
        path = root / "db_table_mapping.json"
        if path.exists():
            import json as _json2
            with path.open("r", encoding="utf-8") as f:
                raw = _json2.load(f)
            if isinstance(raw, dict):
                return set(raw.keys())
    except Exception:
        pass
    return set()


def _quote_catalog_sql_tables(sql: str) -> str:
    """
    Fix catalog SQL for PostgreSQL by quoting table names that are stored uppercase.
    Catalog SQL uses bare names like 'FROM VBRK vk' but PostgreSQL requires 'FROM "VBRK" vk'
    for tables created with quoted uppercase identifiers.

    This replaces unquoted uppercase table names with double-quoted versions throughout the SQL.
    """
    actual_tables = _get_actual_table_names_from_mapping()
    if not actual_tables:
        return sql

    # Only quote tables that are uppercase (lowercase tables like 'vbrp' don't need quotes)
    uppercase_tables = {t for t in actual_tables if t == t.upper() and not t.isdigit()}

    for tbl in sorted(uppercase_tables, key=len, reverse=True):  # longest first to avoid partial matches
        # Match the table name as a standalone word that is NOT already quoted
        # Handles: FROM VBRK, JOIN VBRK, after comma, etc.
        # Does NOT match if already quoted ("VBRK")
        pattern = r'(?<!")\b' + re.escape(tbl) + r'\b(?!")'
        replacement = f'"{tbl}"'
        sql = re.sub(pattern, replacement, sql)

    return sql


def _lookup_sql_catalog(question: str) -> Optional[str]:
    """
    Fast-path: score every catalog entry against the question using keyword matching.
    Returns the best SQL string if a high-confidence match is found, else None.

    Scoring:
      +2 per keyword that appears in the question (whole-word match)
      +1 per keyword that appears anywhere in the question (substring)
      +4 per question_pattern that is a close match (all significant words present)
      +entry.priority bonus
      -3 per neg_keyword that appears in the question

    We only return a match if:
      (a) score >= 5  (strong confidence)
      (b) The question does NOT look like a specific/parametric query
          (i.e. it doesn't contain quoted strings, specific years like 2023/2024,
           or 6+ char non-generic words that look like codes or product names)
    """
    catalog = _get_sql_catalog()
    if not catalog:
        return None

    q_lower = question.lower()
    q_words = set(re.split(r"\W+", q_lower))
    q_words.discard("")

    # ── Detect parametric questions — skip catalog for these ──────────────────
    # If question has quoted strings → specific filter needed, skip catalog
    if re.search(r"['\"]", question):
        return None
    # "containing X", "with description X", "named X" etc. → text-search / specific filter → LLM
    if re.search(
        r"\b(containing|with word|with description|named|called|that include|that has"
        r"|for customer|for vendor|for material|for product)\b",
        q_lower,
    ):
        return None
    # SAP compound codes like DE01, US10, AT03, CC01 → specific entity → LLM
    if re.search(r"\b[A-Z]{1,3}\d{2,4}\b", question):
        return None
    # Country/region-specific filter words (nationality adjectives, "only" with geo noun) → LLM
    # e.g. "Sales by Korean customers only", "revenue from German clients", "US invoices"
    _nationality_adjectives = {
        "korean", "german", "french", "american", "japanese", "chinese", "british",
        "australian", "canadian", "italian", "spanish", "dutch", "swiss", "swedish",
        "norwegian", "danish", "finnish", "polish", "czech", "hungarian", "romanian",
        "portuguese", "greek", "turkish", "indian", "mexican", "brazilian", "russian",
        "thai", "indonesian", "malaysian", "singaporean", "philippine", "vietnamese",
        "south african", "egyptian", "nigerian", "saudi", "emirati", "israeli",
        "austrian", "belgian", "irish",
    }
    for adj in _nationality_adjectives:
        if re.search(r"\b" + re.escape(adj) + r"\b", q_lower):
            return None
    # 2-letter ISO country codes used as filters: "KR", "DE", "US", "GB", etc.
    # Only bypass if they appear as standalone words (not part of another word) and are uppercase
    if re.search(r"\b[A-Z]{2}\b", question):
        # Exclude known SAP/business abbreviations that should NOT trigger bypass
        _ok_codes = {"PO", "GL", "AR", "AP", "GR", "IR", "YY", "AM", "PM", "OK", "ID", "NO"}
        iso_matches = re.findall(r"\b[A-Z]{2}\b", question)
        if any(m not in _ok_codes for m in iso_matches):
            return None
    # "only" as a specificity filter word (e.g. "Korean customers only", "from Germany only")
    # → bypass if "only" appears AND the question has a geographic/specific-entity word
    if "only" in q_words and any(
        kw in q_lower for kw in [
            "country", "region", "city", "customer", "vendor", "product", "material",
            "industry", "sector", "plant", "company", "currency",
        ]
    ):
        return None
    # If question specifies a concrete year (2000-2030) → might need date filter
    # BUT: "by year" or "per year" or "trend by year" is generic — allow it
    year_match = re.findall(r"\b(19\d{2}|20[0-2]\d)\b", question)
    if year_match and not any(p in q_lower for p in ("by year", "per year", "each year", "trend")):
        return None
    # If question contains a specific numeric ID / cost center code (standalone 3–6 digit number)
    if re.search(r"\b\d{3,6}\b", question):
        return None
    # If question contains a title-case proper noun (word starting uppercase mid-sentence)
    # e.g. "sales for customer Siemens" — "Siemens" is a specific name
    # Heuristic: ignore words at the very start; flag if ANY word after position 0 starts uppercase
    # and is NOT a known SAP keyword / acronym
    words_in_question = question.split()
    _known_uppercase = {"SAP","GL","PO","AP","AR","BOM","YoY","KPI","UOM","MRP","ABC","GR","IR",
                        "MARA","MBEW","MARD","MSEG","SKA1","SKAT","MAST","EKBE","KNB1","TCURR",
                        "T001W","T001","T016T","LFA1","LFB1","LFM1","KNA1","BSEG","FAGLFLEXA",
                        "EKKO","EKPO","RBKP","RSEG","VBRK","MAKT","MARC","BSAD","COEP","CSKS",
                        "CEPC","CKIS","KEKO","CKMLCR","KONV","LSEG","LIKP","LIPS","STKO","STPO",
                        "MKPF","RESB","EBAN","AUFK","VBAK","VBAP","VBFA","VBEP","MVKE","KNVV"}
    for w in words_in_question[1:]:  # skip first word (might be a normal capitalised start)
        w_clean = re.sub(r"\W", "", w)
        if (w_clean and w_clean[0].isupper() and not w_clean.isupper()
                and w_clean not in _known_uppercase and len(w_clean) >= 3):
            return None  # title-case proper noun → LLM
    # If question contains a specific material/vendor code pattern (≥6 uppercase chars)
    code_pattern = re.findall(r"\b[A-Z0-9_-]{6,}\b", question)
    _common_abbreviations = {"FAGLFLEXA","EKKO","EKPO","RBKP","RSEG","VBRK","MAKT","MARC",
                              "BSAD","COEP","CSKS","CEPC","CKIS","KEKO","CKMLCR","KONV",
                              "LSEG","LIKP","LIPS","STKO","STPO","MKPF","RESB","EBAN",
                              "AUFK","VBAK","VBAP","VBFA","VBEP","MVKE","KNVV","KNVP",
                              "CRHD","COSP","KEPH","CKHS","CKMLHD","CKMLPP","CKIT",
                              "MARA","MBEW","MARD","MSEG","SKA1","SKAT","MAST","EKBE",
                              "KNB1","TCURR","T001W","T001","T016T","LFA1","LFB1","LFM1",
                              "KNA1","BSEG"}
    specific_codes = [c for c in code_pattern if c not in _common_abbreviations]
    if specific_codes:
        return None

    # ── Score every catalog entry ─────────────────────────────────────────────
    best_score = 0
    best_entry = None

    for entry in catalog:
        score = 0.0

        # Keyword scoring
        for kw in (entry.get("keywords") or []):
            kw_l = kw.lower().strip()
            if not kw_l:
                continue
            kw_words = kw_l.split()
            if len(kw_words) == 1:
                if kw_l in q_words:
                    score += 2          # exact whole-word match
                elif kw_l in q_lower:
                    score += 1          # substring match
            else:
                # multi-word keyword (e.g. "profit center")
                if kw_l in q_lower:
                    score += 3

        # Neg-keyword penalty
        for nk in (entry.get("neg_keywords") or []):
            if nk.lower() in q_lower:
                score -= 3

        # Question-pattern boost: count significant words matched
        for pattern in (entry.get("question_patterns") or []):
            p_words = [w for w in re.split(r"\W+", pattern.lower()) if len(w) >= 4]
            if p_words and all(w in q_lower for w in p_words):
                score += 4
                break  # one pattern match is enough for the boost

        # Priority tie-breaker
        score += (entry.get("priority") or 0) * 0.1

        if score > best_score:
            best_score = score
            best_entry = entry

    if best_score >= 5 and best_entry:
        logger.info(
            "sql_catalog: matched [%s] (score=%.1f) for question: %r",
            best_entry["id"], best_score, question[:80],
        )
        return best_entry.get("sql") or None

    logger.debug(
        "sql_catalog: no confident match (best=%.1f, entry=%s) for: %r",
        best_score,
        best_entry["id"] if best_entry else "none",
        question[:80],
    )
    return None


# --- Static metadata (fallback when mapping/config lack entries) ---------------------------------------------------------------------------

SAP_TABLE_DESCRIPTIONS: Dict[str, str] = {
    # Sales / Billing
    "VBRP": "Billing document item (sales by product, quantities, net values, currencies, customers). Use for: highest sales by product, revenue analysis.",
    "vbrp": "Same as VBRP – billing document item. Use for sales, revenue, product analysis.",
    "VBRK": "Billing document header (invoice-level amounts, dates, currencies, customers).",
    "VBAK": "Sales document header (orders, customers, dates, overall values).",
    "VBAP": "Sales document item (ordered products, quantities, values).",
    "VBEP": "Schedule lines for sales document items (delivery quantities and dates).",
    # Customer / Material Master
    "KNA1": "Customer master (names, addresses, countries, brsch=industry code). IMPORTANT: brsch is a code, use T016T for industry descriptions.",
    "T016T": "Industry text/descriptions (converts brsch codes to readable industry names). Use this for industry labels in charts.",
    "KNVV": "Customer sales data (sales area, pricing, related attributes).",
    "KNVP": "Customer partners (payer, ship-to, bill-to relationships).",
    "MAKT": "Material descriptions (product names).",
    "MARC": (
        "Plant data for material – plant-level material master attributes (MRP, procurement, planning). "
        "Key columns: matnr (material number), werks (plant), "
        "xchar (batch management: 'X'=batch-managed), maabc (ABC indicator: A/B/C), "
        "dismm (MRP type: PD=deterministic, VB=reorder point, etc.), "
        "ekgrp (purchasing group), prctr (profit center), bwtty (valuation category), "
        "plifz (planned delivery time in days), minbe (reorder point qty), "
        "eisbe (safety stock qty), stlan (BOM usage), plnnr (task list number). "
        "Use for: plant-level material parameters, batch-managed materials (WHERE xchar='X'), "
        "ABC classification, MRP settings, profit center assignments by material/plant. "
        "Join: MARC.MATNR = MAKT.MATNR for descriptions."
    ),
    "MARM": "Units of measure for material (UOM conversion). Key columns: matnr, meinh (UOM), umrez (numerator), umren (denominator).",
    "MEAN": "International Article Numbers (EAN/UPC) for materials.",
    "MVKE": "Sales data for materials (sales org, distribution channel, pricing group).",
    # Finance / Accounting
    "BSAD": "Customer open and cleared items (AR line items, payments).",
    "BSEG": "Accounting document segment (line items for GL, customers, vendors).",
    "FAGLFLEXA": (
        "New G/L actual line items – GL cost/revenue postings with profit center, cost center, account. "
        "Key columns: prctr (profit center), racct (GL account), rbukrs (company code), "
        "hsl (amount in LOCAL currency – primary aggregation column), "
        "wsl (transaction currency amount), tsl (transaction currency alternative), "
        "ksl (controlling area currency), osl (object currency), "
        "ryear (fiscal year), gjahr (fiscal year alt), poper (posting period 01-12), "
        "drcrk (debit/credit: S=debit/expense, H=credit/revenue), budat (posting date), "
        "cost_elem (cost element), rcntr (cost center), rtcur (transaction currency – NOT waers), "
        "rwcur (second local currency), belnr (document number), segment (segment). "
        "IMPORTANT: currency is rtcur NOT waers. Amount in local currency is hsl. "
        "For 'total cost by profit center': SELECT prctr, SUM(hsl) FROM FAGLFLEXA GROUP BY prctr. "
        "Use for: GL balances, cost by profit center/cost center/account, P&L analysis."
    ),
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
    "MKPF": (
        "Material document header – goods movement header record. "
        "Key columns: mblnr (document number), mjahr (year), vgart (movement category), "
        "bldat (document date), budat (posting date), usnam (user). "
        "CRITICAL LIMITATION: MKPF is the HEADER only. "
        "MSEG (material document items, which has matnr/quantity/movement type) is NOT in this database. "
        "Therefore: you CANNOT query 'stock movements by material' or 'total quantity issued by material' "
        "using MKPF alone. MKPF can only give you movement counts/dates, NOT quantities per material. "
        "For material-level reservation quantities, use RESB instead."
    ),
    "RESB": (
        "Reservations and dependent requirements – records of reserved material quantities. "
        "Key columns: matnr (material), werks (plant), lgort (storage location), "
        "bdmng (required/reserved quantity), enmng (quantity already withdrawn/issued), "
        "meins (unit of measure), bdter (requirements date), aufnr (order number, join to AUFK), "
        "bwart (movement type), matkl (material group). "
        "Use for: total reserved qty by material (SUM bdmng), issued qty (SUM enmng), "
        "open reservations (bdmng - enmng), reservations by plant, reservations by order. "
        "Join: RESB.MATNR = MAKT.MATNR for material descriptions."
    ),
    "LSEG": "Document segment (document item data).",
    # Pricing / Conditions
    "KONV": (
        "Pricing conditions document – stores condition records for pricing (sales price, "
        "discounts, surcharges). "
        "Key columns: knumv (pricing doc number, join key to VBRK.KNUMV / VBAK.KNUMV), "
        "kschl (condition type, e.g. PR00=standard price, K007=customer discount, RA01=rebate), "
        "kbetr (condition amount/rate), waers (currency), kawrt (condition base value), "
        "kappl (application: V=sales, M=purchasing), kposn (condition item). "
        "Use for: sales price conditions, discount analysis, pricing by material/customer group."
    ),
    # Sales Document Flow
    "VBFA": (
        "Sales document flow – links predecessor and successor documents (order→delivery→billing). "
        "Key columns: vbelv (predecessor doc, e.g. sales order), vbeln (successor doc, e.g. delivery/billing), "
        "vbtyp_n (successor type: J=delivery, M=billing, C=order), posnv (predecessor item), "
        "posnn (successor item), matnr (material). "
        "Use for: tracing order-to-invoice flow, finding all deliveries for an order."
    ),
    # Controlling / Profitability
    "COEP": (
        "CO actual line items – cost postings by cost center, profit center, cost element. "
        "Key columns: kostl (cost center), prctr (profit center), kstar (cost element), "
        "wkgbtr (actual amount in controlling area currency), wtgbtr (amount in transaction currency), "
        "belnr (document number), gjahr (fiscal year), poper (posting period). "
        "Use for: actual cost analysis by cost center or profit center."
    ),
    "COSP": (
        "CO plan totals – planned costs by cost center and cost element. "
        "Key columns: kostl (cost center), kstar (cost element), gjahr (fiscal year), "
        "wkg001..wkg016 (planned amounts per period). "
        "Use for: budget vs actual comparisons, planned cost analysis."
    ),
    "CEPC": (
        "Profit center master data – profit center attributes. "
        "Key columns: prctr (profit center), datbi (valid-to date), kokrs (controlling area), "
        "ktext (short description), ltext (long description), verak (person responsible). "
        "Use for: profit center lookups and labels."
    ),
    "CSKS": (
        "Cost center master data – cost center attributes. "
        "Key columns: kostl (cost center), datbi (valid-to), kokrs (controlling area), "
        "ktext (short text), verak (person responsible). "
        "Use for: cost center lookups and labels."
    ),
    # Product Costing
    "CKIS": (
        "Costing items – detailed cost components for a cost estimate per material. "
        "Key columns: kalnr (costing number, join to KEKO.KALNR), "
        "posnr (item), wertn (total cost value – USE SUM(wertn) for standard cost), "
        "wrtfw (value in foreign currency), kstar (cost element), matnr (material), "
        "kostl (cost center), menge (quantity). "
        "Join pattern: KEKO.matnr → KEKO.kalnr = CKIS.kalnr → SUM(CKIS.wertn) "
        "Use for: material cost breakdown by cost element, standard cost components per material/plant."
    ),
    "CKMLCR": (
        "Material ledger cumulative values – actual (periodic) costs per material/plant. "
        "Key columns: kalnr (join to CKMLHD.kalnr for matnr), bdatj (fiscal year), "
        "poper (posting period), stprs (periodic unit price / standard price), "
        "salk3 (total stock value), waers (currency – this table uses waers), "
        "pvprs (preliminary price). "
        "IMPORTANT: CKMLCR has NO matnr column directly. "
        "To get material: JOIN CKMLHD on CKMLCR.kalnr = CKMLHD.kalnr → use CKMLHD.matnr. "
        "Full join: CKMLCR JOIN CKMLHD ON CKMLCR.kalnr = CKMLHD.kalnr "
        "           JOIN MAKT ON CKMLHD.matnr = MAKT.matnr "
        "Use for: actual cost by material/period, inventory valuation, standard price per period."
    ),
    "CKMLHD": (
        "Material ledger header – identifies the material ledger object (kalnr) per material/plant. "
        "Key columns: kalnr (costing number = join key to CKMLCR/KEKO), "
        "matnr (material number), bwkey (valuation area/plant). "
        "Use as bridge table: CKMLCR.kalnr = CKMLHD.kalnr → CKMLHD.matnr = MAKT.matnr."
    ),
    "KEKO": (
        "Cost estimate header – standard cost estimate per material/plant. "
        "Key columns: matnr (material), werks (plant), kalnr (costing number, join to CKIS.KALNR), "
        "kalka (costing type, '01'=standard cost), kadat (costing date), hwaer (currency – NOT waers), "
        "poper (period), bdatj (year). "
        "NOTE: KEKO itself does NOT have stprs. Standard price is in CKMLCR.stprs (actual) or CKIS.wertn (estimate). "
        "Join KEKO.KALNR = CKIS.KALNR for cost breakdown. "
        "Join KEKO.KALNR = CKMLCR.KALNR for periodic actual costs. "
        "Use for: standard cost lookup, cost estimate headers, material cost by plant."
    ),
    # Internal Orders / Production Orders
    "AUFK": (
        "Order master – internal orders and production orders. "
        "Key columns: aufnr (order number), auart (order type), ktext (description), "
        "kostl (responsible cost center), prctr (profit center), werks (plant). "
        "Use for: order analysis, production order lookups."
    ),
    # Bill of Materials
    "STKO": (
        "BOM header – bill of materials header. "
        "Key columns: stlty (BOM type: M=material), stlnr (BOM number, join to STPO.stlnr), "
        "stlal (alternative BOM), datuv (valid-from date), stktx (description). "
        "NOTE: STKO does NOT have matnr. To link to a material, the MAST table is needed "
        "(material-BOM link), but if MAST is unavailable, query STPO directly. "
        "Use for: BOM header information."
    ),
    "STPO": (
        "BOM items – components in a bill of materials. "
        "Key columns: stlnr (BOM number, join to STKO.stlnr), "
        "idnrk (component material number – join to MAKT.matnr for component description), "
        "menge (component quantity), meins (unit of measure), preis (price), waers (currency). "
        "IMPORTANT: idnrk = the component/child material number. "
        "Join STPO.idnrk = MAKT.matnr to get component descriptions. "
        "Use for: BOM component analysis, what materials go into a product."
    ),
    # AR – Accounts Receivable
    "BSAD": (
        "Customer cleared items (AR) – fully posted AR line items. "
        "Key columns: kunnr (customer, join to KNA1), bukrs (company code), "
        "dmbtr (amount in local currency), wrbtr (amount in transaction currency), "
        "waers (currency), budat (posting date), bldat (document date), "
        "gjahr (fiscal year), belnr (document number), shkzg (debit/credit: S=debit, H=credit). "
        "Use for: AR aging, customer payment analysis, outstanding receivables."
    ),
    # Purchasing Requisition
    "EBAN": (
        "Purchase requisition items – purchase request documents. "
        "Key columns: banfn (requisition number), bnfpo (item), matnr (material), "
        "menge (quantity), meins (UoM), preis (price), waers (currency), "
        "lifnr (preferred vendor), ekgrp (purchasing group), lfdat (delivery date), "
        "erdat (creation date), ebeln (assigned PO number if converted). "
        "Use for: open purchase requisitions, spend request analysis."
    ),
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

TEXT / DESCRIPTION TABLES (IMPORTANT for readable labels):
- KNA1 (customer with brsch code) <-> T016T (industry descriptions)
  * KNA1.brsch = T016T.brsch
  * ALWAYS use T016T.brtxt for industry name (not KNA1.brsch which is just "HITE", "TRAD", "FOOD")

- Materials (MATNR code) <-> MAKT (material text)
  * VBRP.MATNR = MAKT.MATNR or VBAP.MATNR = MAKT.MATNR
  * Use MAKT.MAKTX for product names (not just MATNR codes)

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

PRICING CONDITIONS (KONV):
- KONV (pricing conditions) <-> VBRK (billing header)
  * VBRK.KNUMV = KONV.KNUMV
  * IMPORTANT: You MUST join via VBRK (not VBRP) because KNUMV is on VBRK header, not VBRP item.
  * Typical pattern: SELECT ... FROM vbrp JOIN VBRK ON vbrp.VBELN=VBRK.VBELN JOIN KONV ON VBRK.KNUMV=KONV.KNUMV

- Common KONV.KSCHL condition types (filter by these for specific analysis):
  * 'PR00' = Base price / list price
  * 'K007' = Customer discount (%)
  * 'K004' = Material discount
  * 'RA01' = Customer rebate
  * 'VPRS' = Cost-of-goods-sold (COGS) – use this for margin/profitability!
  * 'MWST'/'MWAS' = Tax
  * 'HD00' = Freight/handling

- For DISCOUNT analysis: filter KONV.KSCHL IN ('K007','K004','RA01') and use KONV.KBETR
- For PRICING: filter KONV.KSCHL = 'PR00' and use KONV.KBETR
- For COGS / COST: filter KONV.KSCHL = 'VPRS' and use KONV.KBETR

- KONV (pricing conditions) <-> VBAK (sales order header)
  * VBAK.KNUMV = KONV.KNUMV

SALES DOCUMENT FLOW (VBFA):
- VBFA.VBELV = predecessor document number (e.g. sales order VBELN)
  * VBFA.VBELN = successor document (delivery or billing)
  * Filter VBFA.VBTYP_N for type: 'J'=delivery, 'M'=billing document, 'C'=order

CONTROLLING / PROFITABILITY (COEP, CEPC, CSKS):
- COEP <-> CSKS (cost center master):  COEP.KOSTL = CSKS.KOSTL
- COEP <-> CEPC (profit center master): COEP.PRCTR = CEPC.PRCTR
- FAGLFLEXA <-> CEPC: FAGLFLEXA.PRCTR = CEPC.PRCTR

MATERIAL DOCUMENT:
- MKPF is the goods movement HEADER only. MSEG (material document items with matnr/quantity) is NOT in this database.
- MKPF CANNOT answer "movements by material" or "quantity issued by material" — use RESB for that.
- MKPF is useful only for: counting movement documents by date, user, or movement category.

RESERVATIONS (RESB):
- RESB <-> MAKT (material name): RESB.MATNR = MAKT.MATNR
- RESB <-> AUFK (order): RESB.AUFNR = AUFK.AUFNR
- RESB.BDMNG = total reserved (required) quantity
- RESB.ENMNG = quantity already withdrawn (issued)
- Open qty = RESB.BDMNG - RESB.ENMNG
- Use: SUM(RESB.BDMNG) for total reserved, SUM(RESB.ENMNG) for total issued, by matnr/werks

PLANT MATERIAL MASTER (MARC):
- MARC <-> MAKT: MARC.MATNR = MAKT.MATNR
- MARC.XCHAR = 'X' means batch-managed material
- MARC.MAABC = ABC indicator (A=high value, B=medium, C=low)
- MARC.DISMM = MRP type (PD=demand-driven, VB=reorder point)
- MARC.PRCTR = profit center assigned to this plant/material

PRODUCT COSTING (STANDARD COST):
- KEKO (cost estimate header) <-> CKIS (costing items)
  * KEKO.KALNR = CKIS.KALNR
  * SUM(CKIS.WERTN) gives total standard cost per material

- KEKO (cost estimate) <-> MAKT (material description)
  * KEKO.MATNR = MAKT.MATNR

MATERIAL LEDGER (ACTUAL COST):
- CKMLCR has NO matnr column. Must join via CKMLHD:
  * CKMLCR.KALNR = CKMLHD.KALNR  (get the material)
  * CKMLHD.MATNR = MAKT.MATNR    (get description)
  * CKMLCR.STPRS = periodic standard price, CKMLCR.SALK3 = stock value
  * Filter by CKMLCR.BDATJ (year), CKMLCR.POPER (period 01-12)

- CKMLHD.BWKEY = plant/valuation area (can filter by plant)

BILL OF MATERIALS (BOM):
- STKO (BOM header) <-> STPO (BOM components)
  * STKO.STLNR = STPO.STLNR  (and STKO.STLTY = STPO.STLTY)

- STPO (component) <-> MAKT (component description)
  * STPO.IDNRK = MAKT.MATNR  (idnrk is the component/child material number)

- NOTE: STKO has NO matnr column. The parent material link requires MAST table which is NOT in DB.
  For "components of material X" queries, you cannot directly filter by parent matnr without MAST.
  Instead, use STPO directly to list components, or query CKIS for cost components.

AR / ACCOUNTS RECEIVABLE:
- BSAD (customer cleared items) <-> KNA1: BSAD.KUNNR = KNA1.KUNNR
- BSEG (accounting line items) <-> KNA1: BSEG.KUNNR = KNA1.KUNNR
- BSAD.DMBTR = amount in local currency, BSAD.SHKZG = S(debit)/H(credit)

PURCHASE REQUISITION:
- EBAN (requisition) <-> LFA1 (vendor): EBAN.LIFNR = LFA1.LIFNR
- EBAN (requisition) <-> EKPO (PO): EBAN.EBELN = EKPO.EBELN AND EBAN.EBELP = EKPO.EBELP

VERY IMPORTANT:
- VBRP / vbrp usually does NOT have KUNNR directly. To reach the customer, go:
  VBRP.VBELN -> VBRK.VBELN, then VBRK.KUNAG -> KNA1.KUNNR.

- When you need INDUSTRY of a customer:
  * NEVER use KNA1.brsch alone (it's just a code like "HITE", "TRAD", "FOOD")
  * ALWAYS join T016T to get the description: T016T.brtxt (readable industry name)
  * Join: KNA1.brsch = T016T.brsch
  * SELECT T016T.brtxt as industry_name (not KNA1.brsch)

- When you need COUNTRY of a customer:
  * PREFERRED: VBRK.LAND1 (country code directly on billing header — no extra join needed)
  * ALTERNATIVE: KNA1.LAND1 via join VBRK.KUNAG = KNA1.KUNNR (only if customer name also needed)

- For cost-related or COGS queries, use EKPO (purchase values), RBKP/RSEG (vendor invoice amounts), BSEG (accounting).

- MARGIN / PROFITABILITY queries:
  * Revenue = SUM(vbrp.NETWR) from billing items
  * COGS option 1 (KONV): JOIN VBRK ON vbrp.VBELN=VBRK.VBELN, JOIN KONV ON VBRK.KNUMV=KONV.KNUMV
    WHERE KONV.KSCHL='VPRS' → SUM(KONV.KBETR) = cost
  * COGS option 2 (purchase cost): JOIN EKPO ON vbrp.MATNR=EKPO.MATNR → SUM(EKPO.NETWR/EKPO.MENGE * vbrp.FKIMG)
  * Gross margin % = (revenue - cost) / revenue * 100
  * Simple margin: SELECT matnr, SUM(netwr) as revenue, ... GROUP BY matnr from vbrp/VBRK

- CURRENCY NOTE (very important):
  * VBRK currency column = WAERK (NOT waers)
  * FAGLFLEXA currency column = RTCUR (NOT waers)
  * KEKO currency column = HWAER (NOT waers)
  * EKKO, RBKP, RSEG, EKPO, KONV, CKMLCR → WAERS (the usual one)
  * Always match the actual currency column name to the table you are querying

- FISCAL YEAR / PERIOD filters:
  * For VBRK/VBRP: use VBRK.FKDAT (billing date, YYYYMMDD format) for date range filters
  * For FAGLFLEXA: use FAGLFLEXA.RYEAR (fiscal year as 4-digit string) and FAGLFLEXA.POPER (period 01-12)
  * For EKKO/RBKP: use BUDAT or BEDAT (YYYYMMDD)
  * For CKMLCR: use BDATJ (year) and POPER (period)
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


def _introspect_columns(db: Session, table_names: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Introspect the DB to get columns for each table and build semantic descriptions.

    Priority:
    1) If db_table_mapping.json (generated by test_db_connection.py) exists, use its
       per-column descriptions for any matching tables/columns.
    2) Otherwise, fall back to simple heuristic descriptions.
    """
    insp = inspect(db.bind)
    mapping: Dict[str, Dict[str, str]] = {}

    # Try to load pre-generated mapping file (optional)
    mapping_file: Dict[str, Any] = {}
    try:
        root = Path(__file__).resolve().parent.parent
        path = root / "db_table_mapping.json"
        if path.exists():
            import json as _json  # local import to avoid polluting module namespace

            with path.open("r", encoding="utf-8") as f:
                raw = _json.load(f)
            if isinstance(raw, dict):
                mapping_file = raw
    except Exception:
        # Non-fatal; we just fall back to heuristics
        mapping_file = {}

    # Build a case-insensitive lookup of actual DB tables once
    db_tables = {t.lower(): t for t in insp.get_table_names()}

    for tbl in table_names:
        # Find matching table in DB, case-insensitive
        actual_name = db_tables.get(tbl.lower())
        if not actual_name:
            # This is critical for debugging questions that reference tables
            # like KONV or FAGLFLEXA that might not actually exist in the DB.
            logger.warning("sap_sql_agent: selected table '%s' is not present in the database", tbl)
            continue

        cols: Dict[str, str] = {}
        file_entry = mapping_file.get(actual_name) or mapping_file.get(tbl) or {}
        file_cols = (file_entry.get("columns") or {}) if isinstance(file_entry, dict) else {}

        for col in insp.get_columns(actual_name):
            col_name = col["name"]
            if isinstance(file_cols, dict) and col_name in file_cols:
                # Use human-friendly description from mapping file when available
                cols[col_name] = str(file_cols[col_name])
            else:
                # Simple heuristic description; LLM will still see raw names
                cols[col_name] = f"{tbl}.{col_name} column"

        if cols:
            mapping[actual_name] = cols

    return mapping


def _get_table_descriptions(db: Session) -> Dict[str, str]:
    """
    Build table descriptions. Priority (mapping-first, adaptive for new tables):
    1) db_table_mapping.json meta.description
    2) schema_ai_config.json table_semantic_hints
    3) SAP_TABLE_DESCRIPTIONS (fallback)
    4) Generic fallback

    Skip tables from schema_ai_config skip_tables.
    """
    insp = inspect(db.bind)
    db_table_names = insp.get_table_names()
    cfg = _get_schema_config()
    skip_set = {s.lower() for s in (cfg.get("skip_tables") or [])}
    db_table_names = [t for t in db_table_names if t.lower() not in skip_set]

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

    table_hints = (cfg.get("table_semantic_hints") or {})
    out: Dict[str, str] = {}
    for actual_name in db_table_names:
        # 1) Mapping meta (from db_table_mapping.json - updated by refresh_schema_for_ai.py)
        entry = mapping_file.get(actual_name) or mapping_file.get(actual_name.upper())
        if isinstance(entry, dict) and entry.get("meta", {}).get("description"):
            desc = str(entry["meta"]["description"])
            if desc and desc != f"{actual_name} table":
                out[actual_name] = desc
                continue
        # 2) Config table_semantic_hints (extensible - add new tables here)
        desc = table_hints.get(actual_name) or table_hints.get(actual_name.upper())
        if desc:
            out[actual_name] = desc
            continue
        # 3) Invoice-bot full descriptions (sap_table_descriptions.json) then hardcoded SAP_TABLE_DESCRIPTIONS
        try:
            from .invoice_bot_helpers import get_table_descriptions
            ib = get_table_descriptions()
            desc = ib.get(actual_name) or ib.get(actual_name.upper())
        except Exception:
            desc = None
        if not desc:
            desc = SAP_TABLE_DESCRIPTIONS.get(actual_name) or SAP_TABLE_DESCRIPTIONS.get(actual_name.upper())
        if desc:
            out[actual_name] = desc
            continue
        # 4) Generic
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

    # ── Helper: get numeric column names from schema config (no hardcoding) ──
    _cfg = _get_schema_config()
    _numeric_col_names = {c.upper() for c in (_cfg.get("numeric_columns") or [])}

    # ── Build a schema-based lookup: which tables have at least one numeric column ──
    # This lets us detect "dimension-only" tables without hardcoding table names.
    def _table_has_numeric_col(tbl_name: str) -> bool:
        """Return True if the table has any column that appears in numeric_columns config."""
        # We need column info from db_table_mapping.json (already in table_descriptions
        # as meta, or we can introspect quickly)
        try:
            from ..database import SessionLocal  # noqa: F401 – only used for quick check
        except Exception:
            pass
        # Use column_mappings from DB introspection cache if available; otherwise
        # check column names we know about from db_table_mapping / SAP_TABLE_DESCRIPTIONS.
        # This is best-effort; if we can't determine, assume the table has numeric columns.
        known_numeric_tables = {
            "VBRP", "VBRK", "EKPO", "EKKO", "RBKP", "RSEG",
            "FAGLFLEXA", "COEP", "KEKO", "CKIS", "BSAD", "BSEG",
            "LIKP", "LIPS", "VBAP", "VBAK",
        }
        return tbl_name.upper() in known_numeric_tables

    q_lower = (question or "").lower()
    db_tables_lower = {t.lower(): t for t in table_descriptions}

    # ── STEP 1: Detect if user explicitly named any tables in the question ──
    # e.g. "from FAGLFLEXA", "using KNA1 and T016T", "KONV pricing conditions"
    explicit_tables: List[str] = []
    for tbl_name in table_descriptions.keys():
        name_lower = tbl_name.lower()
        if name_lower and len(name_lower) >= 3 and name_lower in q_lower:
            explicit_tables.append(tbl_name)

    if explicit_tables:
        logger.info("sap_sql_agent: user explicitly named tables: %s", explicit_tables)

        # ── STEP 1a: Detect if this is a pure listing/filter query ────────────
        # Pure listing queries (e.g. "show all products containing 'jacket' from MAKT",
        # "list all customers in Germany") do NOT need fact/transaction tables.
        # Only add fact tables when the question asks for aggregation/amounts.
        _aggregation_signals = (
            "total", "sum", "revenue", "sales", "spend", "cost", "amount",
            "how much", "count", "how many", "average", "avg",
            "highest", "lowest", "top", "best", "worst", "most", "least",
            "maximum", "minimum", "by customer", "by product", "by country",
            "by vendor", "by material", "margin", "profit", "price",
            "ranking", "rank", "compare",
        )
        _is_aggregation_query = any(sig in q_lower for sig in _aggregation_signals)

        # ── STEP 1b: Schema-driven fact-table enrichment ──────────────────────
        # Only enrich with fact tables if the question clearly needs aggregation.
        # For pure listing/filter queries (containing, with word, list/show + single table),
        # the dimension table alone is sufficient — do NOT add VBRP/VBRK unnecessarily.
        has_fact_table = any(_table_has_numeric_col(t) for t in explicit_tables)
        if not has_fact_table and _is_aggregation_query:
            # Ask the LLM to identify missing fact tables given the question + named tables
            enrich_prompt = f"""
The user asked: "{question}"

They explicitly named these database tables: {explicit_tables}

These tables are dimension/lookup tables with no financial amounts.
Given the question, which FACT/TRANSACTION tables from the list below are needed
to actually compute the answer?

If the question is ONLY asking to LIST or FILTER records (e.g. "show all products containing X",
"list all customers with Y") and does NOT need totals/sums/counts, return:
{{"fact_tables": []}}

Otherwise return the needed fact tables:
{{"fact_tables": ["<exact_table_name>", ...]}}

Available tables:
{json.dumps(table_descriptions, indent=2)}
"""
            try:
                er = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": enrich_prompt}],
                    temperature=0,
                )
                ec = er.choices[0].message.content or ""
                try:
                    ed = json.loads(ec)
                except json.JSONDecodeError:
                    m2 = re.search(r"\{.*\}", ec, re.DOTALL)
                    ed = json.loads(m2.group(0)) if m2 else {}
                fact_tables = [
                    db_tables_lower.get((t or "").strip().lower())
                    for t in ed.get("fact_tables", [])
                    if isinstance(t, str) and db_tables_lower.get((t or "").strip().lower())
                ]
                for ft in fact_tables:
                    if ft and ft not in explicit_tables:
                        explicit_tables.insert(0, ft)
                logger.info(
                    "sap_sql_agent: added fact tables %s to explicit list", fact_tables
                )
            except Exception as enrich_err:
                logger.warning("sap_sql_agent: fact-table enrichment LLM call failed: %s", enrich_err)
        elif not has_fact_table and not _is_aggregation_query:
            logger.info(
                "sap_sql_agent: pure listing/filter query — using dimension table(s) %s as-is (no fact-table enrichment)",
                explicit_tables,
            )

        return explicit_tables

    # ── STEP 2: Full LLM-driven table selection ───────────────────────────────
    # No hardcoded signal words — the prompt is comprehensive enough to handle
    # ANY question type: revenue, cost, procurement, logistics, GL, etc.
    knowledge_block = ""
    if knowledge_context and knowledge_context.strip():
        knowledge_block = f"""
User preferences / stored knowledge (apply when relevant):
{knowledge_context.strip()}
"""

    prompt = f"""
User question: "{question}"
{knowledge_block}
You are selecting SAP-style database tables needed to answer this business question.

Available tables (use EXACT names):
{json.dumps(table_descriptions, indent=2)}

=== MANDATORY SELECTION RULES ===

1. FACT / TRANSACTION TABLES — always include the table(s) that actually hold the numbers:
   - Sales, billing, revenue, turnover, income → VBRP + VBRK (always both)
   - Purchase orders, procurement, ordered quantity, purchase cost → EKKO + EKPO
   - Vendor invoices, vendor spend, accounts payable, supplier payment, total spend by vendor,
     highest spend vendor, AP, invoice amount → RBKP + RSEG (RBKP = header, RSEG = line items)
   - General ledger, profit center accounting, FI postings, GL balance → FAGLFLEXA
   - Deliveries, shipments, logistics → LIKP + LIPS
   - CO actual costs by cost center → COEP + CSKS
   - Standard cost / unit cost estimates → KEKO (+ CKIS for detail breakdown)
   - Pricing conditions, discounts, surcharges → KONV + VBRK (join on KNUMV)
   - Material reservations, reserved quantity, issued quantity → RESB
   - Plant-level material parameters, MRP, ABC classification → MARC
   - Internal orders, project orders → AUFK + COEP
   - AR receivables, overdue items, payment clearing → BSAD + BSEG

2. DIMENSION / LOOKUP TABLES — always add these alongside the fact tables:
   - Any question involving customers, buyers, sold-to parties → KNA1
   - Any question involving materials, products, items → MAKT
   - Any question involving vendors, suppliers → LFA1
   - Any question explicitly about "industry" or "sector" → T016T (ONLY with KNA1)
   - Do NOT include T016T unless the question explicitly mentions industry/sector —
     T016T only has brsch and brtxt columns, no financial data.

3. COST-OF-PRODUCT rule:
   - "What is the cost / price of [product]?" or "unit cost" or "standard cost" →
     use KEKO + MAKT (join KEKO.MATNR = MAKT.MATNR)
   - Do NOT use EKPO for unit cost (EKPO = bulk purchase orders, not unit standard costs)

4. VENDOR SPEND rule (very important):
   - "Highest spend by vendor", "top vendors by invoice", "total vendor spend",
     "vendor invoice totals" → RBKP + LFA1 (LFA1 = vendor name)
   - RBKP.rmwwr = invoice amount (gross), RBKP.lifnr = vendor number (join LFA1.lifnr)
   - GROUP BY LFA1.name1, SUM(RBKP.rmwwr) ORDER BY total_spend DESC

5. INVOICE LISTING rule (show/list individual invoice rows with names):
   - "Show invoices with payer/customer names" → VBRK + KNA1
   - "Show invoices with payer/customer names by industry" → VBRK + KNA1 + T016T
   - "Show billing documents with customer details" → VBRK + KNA1
   - "List all invoices with [customer/payer/billing] info" → VBRK + KNA1
   - These are ROW-LEVEL queries (individual invoice rows), NOT aggregated.
   - Do NOT include VBRP for these (VBRK has the header; VBRP adds complexity).

6. When the user explicitly names specific tables in the question (e.g. "using KONV",
   "from FAGLFLEXA", "using EKPO"), ALWAYS use exactly those tables — do not substitute.

7. Choose the MINIMUM set of tables. Do not include tables unrelated to the question.

Return STRICT JSON only — no explanation:
{{
  "selected_tables": [
    {{ "name": "<exact_table_name>", "reason": "<one line why>" }}
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

    tables = [
        t.get("name") for t in data.get("selected_tables", [])
        if isinstance(t, dict) and t.get("name")
    ]

    # Normalize: keep only tables that actually exist in the DB
    normalized = []
    for t in tables:
        key = (t or "").strip()
        if not key:
            continue
        actual = db_tables_lower.get(key.lower())
        if actual and actual not in normalized:
            normalized.append(actual)
    tables = normalized

    # ── STEP 3: Structural safety rules (no semantics, just DB constraints) ──
    # These are schema-structural facts, not signal-word heuristics:

    # T016T has only brsch+brtxt — joining it without KNA1 makes no sense
    if any(t.upper() == "T016T" for t in tables) and not any(t.upper() == "KNA1" for t in tables):
        tables = [t for t in tables if t.upper() != "T016T"]
        logger.info("Removed T016T: it needs KNA1 as parent but KNA1 was not selected")

    # Last-resort fallback: if LLM returned nothing, start with VBRP
    if not tables:
        for cand in ["VBRP", "vbrp", "VBRK"]:
            actual = db_tables_lower.get(cand.lower())
            if actual:
                tables = [actual]
                break
        if not tables and table_descriptions:
            tables = [list(table_descriptions.keys())[0]]

    logger.info("sap_sql_agent: final selected tables: %s", tables)
    return tables


# --- Adaptive (invoice-bot-style) path: richer prompts for table selection and SQL spec ---

def _safe_json_extract_adaptive(text: str) -> Dict[str, Any]:
    """Parse JSON from LLM response; tolerate wrapped text."""
    if not text or not text.strip():
        return {}
    s = text.strip()
    for attempt in range(2):
        if attempt == 1:
            m = re.search(r"\{[\s\S]*\}", s)
            if m:
                s = m.group(0)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            continue
    return {}


def _enrich_tables_by_intent(question: str, selected_tables: List[str]) -> List[str]:
    """
    Adaptively ensure selected_tables include tables for the question's intent.
    Uses intent tokens so any wording (e.g. 'cost of jacket-related postings by profit center')
    gets the right tables without hardcoding exact phrases.
    """
    if not (question or "").strip():
        return selected_tables
    q = (question or "").lower()
    intents = _get_query_intent_tokens(question)
    tables = list(selected_tables) if selected_tables else []
    tables_upper = {t.upper() for t in tables}

    # Cost / balance by profit center (any phrasing) -> need FAGLFLEXA
    if ("profit_center" in intents or "cost_center" in intents) and ("cost" in intents or "amount" in q or "balance" in q or "posting" in q):
        if "FAGLFLEXA" not in tables_upper:
            tables.insert(0, "FAGLFLEXA")
            tables_upper.add("FAGLFLEXA")
    # Product/material/jacket in question with cost or profit center -> need MAKT for description/filter
    if "material" in intents or "product" in intents or any(w in q for w in ("jacket", "harley", "product", "material")):
        if "MAKT" not in tables_upper and ("FAGLFLEXA" in tables_upper or "cost" in intents or "profit_center" in intents or "revenue" in intents):
            tables.append("MAKT")
            tables_upper.add("MAKT")
    # Revenue/sales/customer -> ensure VBRK, VBRP, KNA1 when relevant
    if ("revenue" in intents or "sales" in intents) and "customer" in intents and "VBRK" not in tables_upper:
        tables.extend(["VBRK", "VBRP", "KNA1"])
    return tables


def _get_query_intent_tokens(question: str) -> set:
    """
    Extract intent tokens from a natural language query for dynamic table fallback.
    Maps synonyms and common phrasings to canonical intents so any wording is handled.
    """
    if not (question or "").strip():
        return set()
    q = (question or "").lower()
    # Normalize: replace common synonyms so "sold" -> sales, "spend" -> cost, etc.
    synonyms = [
        ("revenue", "sales", "billing", "invoices", "sold", "sell", "sale", "billed", "invoice", "revenues"),
        ("cost", "costs", "spend", "spending", "expense", "expenses", "amount", "amounts", "price", "prices", "posting", "postings"),
        ("purchase", "purchased", "buy", "bought", "procure", "procurement", "po ", "pos "),
        ("customer", "customers", "client", "clients", "buyer", "buyers"),
        ("vendor", "vendors", "supplier", "suppliers"),
        ("profit center", "profit centre", "profitcenter", "prctr"),
        ("cost center", "cost centre", "csks", "costcenter"),
        ("delivery", "deliveries", "delivered", "ship", "shipped", "shipment"),
        ("margin", "margins", "profit", "profitability", "gross margin"),
        ("industry", "industries", "sector", "brsch"),
        ("material", "materials", "product", "products", "item", "items", "sku"),
        ("quantity", "quantities", "qty", "volume"),
        ("year", "years", "annual", "fiscal", "yoy", "y/y"),
        ("top ", "best ", "highest ", "largest ", "biggest ", "leading "),
        ("total ", "sum ", "aggregate", "breakdown", "by "),
    ]
    tokens = set()
    for canonical_group in synonyms:
        group = canonical_group if isinstance(canonical_group, tuple) else (canonical_group,)
        key = group[0]
        for term in group:
            if term in q:
                tokens.add(key.replace(" ", "_"))
                break
    # Single-word intents
    if any(x in q for x in ("vbrk", "vbrp", "ekpo", "ekko", "makt", "mara", "kna1", "faglflexa", "likp", "lips")):
        tokens.add("table_mentioned")
    if any(x in q for x in ("how much", "what is", "what are", "show me", "give me", "list ", "get ", "tell me")):
        tokens.add("ask_value")
    return tokens


def _pick_tables_adaptive(
    question: str,
    client: OpenAI,
    db: Session,
    knowledge_context: Optional[str] = None,
) -> List[str]:
    """
    Invoice-bot-style table selection: same rules (product-only, revenue, delivery,
    credit, costing, process flow) so each query gets the right tables.
    Returns list of table names.
    """
    table_descriptions = _get_table_descriptions(db)
    if not table_descriptions:
        table_descriptions = SAP_TABLE_DESCRIPTIONS

    prompt = f"""
Interpret the user's intent flexibly. Treat paraphrases and synonyms as equivalent:
- revenue = sales = billing = invoices = sold = billed; cost = spend = expense = amount;
- purchase = bought = procured; customer = client; vendor = supplier;
- "how much" / "what is" / "show me" / "list" / "give me" = request for data;
- "by" = "per" = "for each" = breakdown dimension. Never return empty selected_tables just because wording is informal or different.

User query: "{question}"

Available tables:
{json.dumps(table_descriptions, indent=2)}

Task:
- Identify tables relevant to answer the query.
- If the query involves customer number or customer name, always include KNA1 (customer number = KUNNR, customer name = NAME1).
- For value determination, revenue, or "best products by value": prefer VBRK (header) and VBRP (item: NETWR, MATNR); join on VBELN. Revenue is shown only for billing category types A, B, C, D, E, I, L, W (VBRK.FKTYP).
- For "best products" or "products by value and industry": use VBRK and VBRP for value; add KNA1 for industry (BRSCH); add MAKT and join MAKT.MATNR = VBRP.MATNR so results show material names (MAKTX).
- **Product/master data only (no revenue, no logistics):** When the user asks to show/list/filter **products by name only** (e.g. "show Harley products") and does NOT ask for revenue, sales value, invoices: use ONLY **master data tables** — MARA, MAKT, MEAN, MARM, MVKE. Do NOT use VBRK, VBRP, VBAK, VBAP, BSAD, BSID or LIKP, LIPS, VBFA, VTTK.
- **Revenue / sales value (when explicitly asked):** When the user asks for revenue, sales value, net value, invoices: use VBRK, VBRP, VBAK, VBAP, BSAD, BSID.
- **Delivery / logistics (when explicitly asked):** LIKP, LIPS, VBFA, VTTK only when the user explicitly asks for delivery, shipments, logistics.
- For product attributes, material master: include MARA and MAKT; join MARA.MATNR and MAKT.MATNR to VBRP.MATNR when combining with sales data.
- For industry trends: use VBRK, VBRP, KNA1 (BRSCH), MAKT (MAKT.MATNR = VBRP.MATNR).
- Sales order data: use VBAK (header) and VBAP (item). For billing documents: include VBRK, VBRP, join to VBAK (VBRP.AUBEL = VBAK.VBELN).
- Delivery-specific: when the user asks about delivery or process flow: include LIKP, LIPS, VBFA.
- For product costing, cost of goods, standard price: use MBEW (STPRS, VERPR, VPRSV, PEINH), KEKO, KEPH, MARA, MAKT.
- For purchase orders: use EKKO, EKPO, LFA1 (vendor), MARA, MAKT.
- **"Show me last sales" / "best sales" / "recent sales" / "last best sales" / "top sales"**: always select VBRK, VBRP, and KNA1 (billing documents and customer). Do not return empty selected_tables.
- **Sales/revenue by country or "X customers only"** (e.g. "Sales by Korean customers only", "revenue from India", "German customers"): use VBRK, VBRP, and KNA1 (customer country = KNA1.LAND1; or use VBRK.LAND1). Always include these tables so the query can filter by country code (e.g. KR, IN, DE).
- **Sales by year / revenue by year / total sales per year**: use VBRK and VBRP (billing header and item). Group by year from VBRK.FKDAT or VBRK.GJAHR. Do not require a specific table name in the question.
- **Cost by profit center, cost by GL account, cost by profit center and GL account, cost by profit center and GL account for last N months**: use FAGLFLEXA only (columns: prctr=profit center, racct or cost_elem=GL account, hsl=amount in local currency, ryear, poper, budat for date). Do NOT use EKPO, RBKP, RSEG, or KEKO for profit center or GL account breakdowns.
- **Any profit center cost/balance question** (total balance by profit center and fiscal year, top N profit centers by cost, monthly cost trend, segment, rcntr, company code, last fiscal year, unusually high costs, average cost per transaction, partner profit center pprctr, functional area rfarea): use FAGLFLEXA; add VBRK/VBRP/KNA1/MAKT only when the question explicitly asks to link costs to customers or products.
- **Link FAGLFLEXA profit center costs back to customers/products / profit center and customer / cost by profit center and customer**: use FAGLFLEXA with VBRK, VBRP, KNA1, MAKT when the question asks to link or attribute costs to customers or products. Select FAGLFLEXA (prctr, hsl, racct, ryear, poper), VBRK/VBRP (revenue/customer), KNA1 (customer name), MAKT (material name). Join where document or segment allows; if no direct join in schema, still return FAGLFLEXA by profit center and optionally by cost element so the user gets cost breakdown.
- **Purchasing (EKPO, quantity, cost, material, plant, vendor)**: use EKKO, EKPO; add MAKT (MATNR description), MARA, LFA1 (vendor) when question asks for material name, vendor, or plant. For "jacket" or product name filter use MAKT and filter MAKT.MAKTX.
- **Stock movements, reservations (MKPF, RESB, MSEG)**: use MKPF (header), MSEG or RESB as needed; join to MARA/MAKT for material names.
- **Plant/master (MARC, on-hand, MRP, slow-moving)**: use MARC, MARA, MAKT when question asks plant-level data, on-hand, MRP parameters, or slow-moving materials.
- **Costing (KEKO, CKMLCR, CKIS, standard cost, BOM)**: use KEKO, KEPH, CKIS for cost breakdown; CKMLCR for period totals/stock value; STKO, STPO, MAST for BOM/component questions.
- **Pricing/conditions (KONV)**: use KONV with VBRK (KNUMV), VBRP, MAKT when question asks for prices, discounts, conditions, PR00, or list price.
- **Revenue by industry (T016T)**: use KNA1 (BRSCH) and T016T (join KNA1.brsch = T016T.brsch) for industry description; use with VBRK, VBRP.
- **Customer group / sales area (KNVV)**: use KNVV with KNA1, VBRK, VBRP when question asks customer group (kdgrp), sales org, or distribution channel.
- **AR, receivables (BSAD, BSEG)**: use BSAD, BSEG, BKPF with KNA1 when question asks open AR, aging, credit, write-offs, or payment terms.
- **Vendors, AP (LFA1, RBKP, RSEG)**: use LFA1 (vendor master), EKKO, EKPO for PO spend; RBKP, RSEG for invoice amounts; LFB1 for payment terms.
- **Margin / profitability (revenue minus cost)**: use VBRK, VBRP (revenue) with EKPO or RSEG (cost) and MAKT; join on material where possible.
- **Improving margins / margin year over year / products with improving margins**: use VBRK, VBRP (revenue), EKPO (cost), MAKT; group by material and year (GJAHR or FKDAT) to show margin trend.
- **Compare costs between profit centers / two profit centers**: use FAGLFLEXA only; select prctr, SUM(hsl); group by prctr.
- **Deliveries (LIKP, LIPS)**: use LIKP, LIPS, VBFA with VBRK, VBRP when question asks delivered quantity, on-time delivery, or delivery performance.
- **Controlling (AUFK, COEP, COSP, CSKS, CEPC)**: use AUFK (internal orders), COEP/COSP (actual/planned cost), CSKS (cost center), CEPC (profit center master) when question asks cost by order, cost center, or profit center master.
- **Top N / top 20 / top 10**: when question asks "top N customers/materials/vendors/documents" use the relevant tables (VBRK/VBRP/KNA1 for customers/materials, EKPO/EKKO/LFA1 for vendors, LIKP/LIPS for deliveries, FAGLFLEXA for profit centers). Always return at least one table.
- **Jacket / Harley / product by name**: when question mentions "jacket", "Harley", or a product name, include MAKT (and VBRP/EKPO as needed); join MAKT.MATNR to show material description; filter MAKT.MAKTX for the name.
- **Revenue by customer and year**: VBRK, VBRP, KNA1; group by customer (KUNNR or NAME1) and GJAHR or FKDAT year.
- **Revenue by industry (T016T)**: VBRK, VBRP, KNA1, T016T; join KNA1.brsch = T016T.brsch.
- **Revenue by customer group (KNVV.kdgrp)**: VBRK, VBRP, KNA1, KNVV; join on KUNNR.
- **Revenue by sales org / distribution channel (vkorg, vtweg)**: VBRK, VBRP; group by VBRK.vkorg, VBRK.vtweg.
- **Incoterms (inco1, inco2)**: VBRK, VBRP; include VBRK.inco1, inco2.
- **Standard cost / KEKO / CKMLCR / CKMLPP**: use KEKO, KEPH, CKIS for cost breakdown; CKMLCR for period/stock value (stprs, salk3); CKMLPP for variances. Add MARA, MAKT for material name.
- **BOM (STKO, STPO, MAST)**: use STKO, STPO, MAST, MARA, MAKT when question asks components, BOM, explosion, or "Harley jacket BOM".
- **Cost by cost center (CSKS, COEP)**: use COEP, CSKS (cost center master); join COEP to CSKS; CEPC for profit center master.
- **Internal orders (AUFK)**: use AUFK, COEP, COSP when question asks internal order, order cost, or project.
- **CKMLPP (variances)**: use CKMLPP with CKMLCR, MARA, MAKT when question asks planned vs actual cost variance or material variances.
- **MVKE (material hierarchy, prodh, region)**: use MVKE with VBRK, VBRP, MAKT when question asks revenue by material hierarchy or product hierarchy or region.
- **KNVP (partner roles)**: use KNVP with KNA1 when question asks payer, bill-to, ship-to, or customer partner relationships.
- **VBAP, VBEP (open quantity, backorder, schedule)**: use VBAP, VBEP with LIPS, VBRP when question asks open vs delivered quantity, backorder, or schedule adherence.
- **Write-offs, credit notes, blocking**: use BSAD, BSEG, KNA1 (and BKPF if available) for write-offs by customer/year; include when question asks credit notes, returns, or blocking.
- **AR by profit center**: use BSAD/BSEG with account assignment fields or FAGLFLEXA when question asks AR balances by profit center.
- **Cost of a product (jacket, Harley)**: use EKPO, EKKO, MAKT for purchase cost; or KEKO, KEPH, MAKT for standard cost when question says "standard cost" or "costed".
- **Cost of jacket-related postings by profit center (or any "cost/postings/balance by profit center" with a product word)**: Always use FAGLFLEXA for the cost/postings; add MAKT (and optionally VBRP) when the question mentions a product (jacket, Harley, material). If the schema does not link FAGLFLEXA to materials, still return FAGLFLEXA cost grouped by profit center (prctr). Never return empty tables for this intent.
- **Purchase order totals by material**: use EKKO, EKPO, MAKT; group by MATNR; SUM(NETWR) or SUM(MENGE) as totals; add MAKT for material name.
- **Compare sales data with invoice data**: use VBRK, VBRP (billing/sales) and RBKP, RSEG (vendor invoices) or same billing as "invoice"; KNA1 if by customer. Return comparison (e.g. by document, by amount, or side-by-side).
- **Compare 2023 vs 2024 sales (or two years)**: use VBRK, VBRP; filter GJAHR or year from FKDAT in (2023, 2024); group by year; SUM(NETWR) per year for comparison.
- **Revenue last 30 days / Sales for 2024**: use VBRK, VBRP; add date filter FKDAT >= current_date - 30 or FKDAT in 2024.
- **Materials in sales but not in purchasing (or vice versa)**: use VBRP and EKPO (and MARA, MAKT) to compare material lists; LEFT JOIN and WHERE NULL for "not in" logic.
- When in doubt, prefer including tables that might be relevant (e.g. VBRK+VBRP+KNA1 for anything about sales/customers/revenue) so the next step can refine the query. Prefer a reasonable answer over returning no tables.
- Only return JSON in this format:

{{
  "query": "...",
  "selected_tables": [
    {{ "name": "TABLE_NAME", "description": "..." }}
  ]
}}
"""
    if knowledge_context:
        prompt += f"\nUser context / preferences: {knowledge_context}\n"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = (resp.choices[0].message.content or "").strip()
        out = _safe_json_extract_adaptive(content)
        if not out or "selected_tables" not in out:
            return []
        names = [t.get("name") for t in out["selected_tables"] if t.get("name")]
        # Ensure we only return tables that exist in table_descriptions (or known SAP tables)
        return [n for n in names if n]
    except Exception as e:
        logger.warning("_pick_tables_adaptive failed: %s", e)
        return []


def _generate_sql_json_adaptive(
    question: str,
    selected_tables: List[str],
    column_mappings: Dict[str, Dict[str, str]],
    client: OpenAI,
    time_scope: str = "current",
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
    table_descriptions: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Invoice-bot-style SQL JSON spec generation: same rules for customer, revenue,
    process flow, filters, order_by, so each query gets the right columns and joins.
    """
    tbl_desc = table_descriptions or {}
    if not tbl_desc and selected_tables:
        tbl_desc = {t: SAP_TABLE_DESCRIPTIONS.get(t, f"Table {t}") for t in selected_tables}

    date_instruction = ""
    if time_scope == "historical":
        date_instruction = "MUST add date filters for 1994-01-01 to 2010-12-31."
    elif time_scope == "current":
        date_instruction = "Include recent data; add date filter only if user specifies a period."
    else:
        date_instruction = "Include ALL periods for comparison."

    few_shot_block = ""
    if few_shot_examples:
        cleaned = [
            {"user_query": str(e.get("user_query") or "")[:400], "sql_query": str(e.get("sql_query") or "")[:800]}
            for e in few_shot_examples[:4] if e.get("user_query") and e.get("sql_query")
        ]
        if cleaned:
            few_shot_block = "\nRecent question→SQL examples (use as patterns):\n" + json.dumps(cleaned, indent=2)

    tables_block = {t: tbl_desc.get(t, f"Table {t}") for t in selected_tables}
    semantic_block = _get_semantic_prompt_block()

    prompt = f"""
Interpret the user's intent flexibly: revenue/sales/billing/invoices mean the same; cost/spend/amount/expense mean the same; infer the correct columns and joins from context even if the user used informal or partial wording. Always return a valid spec when the tables can answer the question.
{semantic_block}User query: "{question}"

Tables available:
{json.dumps(tables_block, indent=2)}

Column mappings (table -> column -> description):
{json.dumps(column_mappings, indent=2)}

**Time scope:** {date_instruction}
{few_shot_block}

**Customer:** Use KNA1.KUNNR (customer number) and KNA1.NAME1 (customer name). Join VBRK.KUNAG = KNA1.KUNNR.
**Revenue:** Use VBRK, VBRP; join VBRK.VBELN = VBRP.VBELN. VBRP has NETWR, MATNR. Revenue only for billing types A,B,C,D,E,I,L,W (FKTYP).
**"Show me last sales" / "best sales" / "recent sales" / "last best sales including year":** You MUST return a valid spec. Use tables VBRK and VBRP (and KNA1 if customer name is needed). Columns: VBRK.VBELN (billing_doc), VBRK.FKDAT (billing_date), VBRK.GJAHR (year – include when user asks for "including year"), VBRK.KUNAG or KNA1.NAME1 (customer), VBRP.NETWR (amount, agg null for row-level or SUM for totals). Joins: VBRP to VBRK on VBELN; VBRK to KNA1 on KUNAG=KUNNR. order_by: VBRP.NETWR DESC or VBRK.FKDAT DESC. limit 100. Never return empty columns or tables.
**Country filter (Korean/Indian/German customers, revenue from India, etc.):** Add filter LAND1 = '<ISO code>': Korean→KR, Indian/India→IN, German/Germany→DE, US→US, UK→GB. Use KNA1.LAND1 when KNA1 is in the query (customer country), or VBRK.LAND1 when only VBRK is used. The system will inject this from the question if you omit it.
**Sales by year / revenue by year:** Use VBRK.GJAHR (fiscal year) as the year dimension. Add column VBRK.GJAHR with description "year"; add group_by VBRK.GJAHR; select SUM(VBRP.NETWR) as total_sales. If GJAHR is not in the mappings, use SUBSTRING(VBRK.FKDAT::text, 1, 4) as year and group by it.
**Cost by profit center and GL account (or "cost by profit center", "cost by GL account"):** Use table FAGLFLEXA only. Select prctr (profit center), racct or cost_elem (GL account), SUM(hsl) as total_cost or total_amount. Add group_by prctr and racct (or cost_elem). For "last 24 months" filter on ryear and poper (or budat) to restrict to recent periods; use current year and prior year with poper 01-12.
**Best products by value and industry:** Use VBRK, VBRP, KNA1 (BRSCH), MAKT (MAKT.MATNR = VBRP.MATNR). Select MATNR, MAKTX, BRSCH, NETWR. Order by NETWR DESC.
**Highest sales by customer:** Use only VBRK, VBRP, KNA1; select KUNNR, NAME1, NETWR; do NOT add VBAK, VBFA, LIKP, LIPS. Order by NETWR DESC.
**Sales by currency (e.g. total sales by WAERS/WAERK):** Use VBRK (WAERK) and VBRP (NETWR). Group by VBRK.WAERK; select WAERK, SUM(NETWR) as total_sales.
**Process flow (billing, order, delivery):** Include billing doc (VBRK.VBELN), sales order (VBRP.AUBEL or VBAK.VBELN), delivery (LIKP.VBELN via VBFA), purchase order (VBAK.BSTNK). Join VBRP.AUBEL = VBAK.VBELN; VBRP to VBFA to LIKP.
**Product by name filter:** When user asks for products matching a name (e.g. "Harley"): add filter MAKT.MAKTX with operator "=" and rhs the product name; use MAKT.SPRAS = 'E' for one language.
**Cost of a product:** Use MBEW, KEKO, KEPH, MARA, MAKT. Select STPRS, VERPR, PEINH, VPRSV, MAKTX. Filter by MAKT.MAKTX for product name.
**Link FAGLFLEXA to customers/products:** Return FAGLFLEXA columns (prctr, racct or cost_elem, hsl, ryear, poper). If VBRK/VBRP/KNA1/MAKT are in tables, add them: join FAGLFLEXA to billing/customer where schema allows (e.g. document or segment); select prctr, SUM(hsl) as total_cost, and customer/material name when available. If no join exists, return FAGLFLEXA grouped by prctr (and racct) with SUM(hsl). Never return empty columns.
**Purchasing (EKPO):** Use EKKO, EKPO; join EKPO.EBELN = EKKO.EBELN. Select MATNR, WERKS (plant), NETPR, MENGE, and SUM for totals; add MAKT (MAKT.MATNR = EKPO.MATNR, SPRAS='E') for material name. Filter MAKT.MAKTX ILIKE '%jacket%' when user asks jacket products. Group by material/plant/vendor as needed.
**Stock movements (MKPF, MSEG, RESB):** Use MKPF (BLDAT, BUDAT), MSEG (MATNR, MENGE, BWKEY) join on MBLNR/MJAHR; or RESB for reservations. Join MARA/MAKT for material name. Group by material; SUM(quantity) for totals.
**Revenue by industry:** Join KNA1.BRSCH = T016T.BRSCH; select T016T text (brtxt) as industry, SUM(VBRP.NETWR). Use VBRK, VBRP, KNA1, T016T.
**Pricing (KONV):** Join KONV to VBRK on KNUMV; use KSCHL (condition type), KWERT or KBETR, KWAER. Add VBRP, MAKT for material-level conditions. Filter MAKT.MAKTX for product name when asked.
**AR / receivables (BSAD, BSEG):** Use BSAD (cleared), BSEG (line items); join to BKPF on BELNR/BUKRS/GJAHR; join KNA1 on KUNNR. Select customer, amount, clearing date; group by customer for totals.
**Vendors (LFA1, RBKP, RSEG):** Use LFA1 (LIFNR, NAME1), EKPO/EKKO for PO spend; RBKP (invoice header), RSEG (invoice item) for invoice amounts. Join on LIFNR, document keys. Sum by vendor, material, or year.
**Margin:** Select VBRP.NETWR (revenue), EKPO.NETWR or RSEG amount (cost); join VBRP.MATNR = EKPO.MATNR where possible. Compute margin = revenue - cost; group by material or customer.
**Improving margins / margin year over year:** Use VBRK, VBRP (revenue by material, year via GJAHR or FKDAT), EKPO or RSEG (cost). Group by material (MATNR) and year; compute margin = SUM(revenue) - SUM(cost) per year. For "improving" or "year over year" return material, year, revenue, cost, margin so the user can see trend; or filter to materials where margin in latest year > prior year. Add MAKT for material name (MAKT.MATNR = VBRP.MATNR).
**Deliveries (LIKP, LIPS):** Join LIKP to LIPS on VBELN; join to VBRP/VBFA for value. Select delivery doc, customer, material, quantity, value. Order by quantity or value DESC.
**Controlling (AUFK, COEP, COSP, CSKS):** Use COEP for actual cost by cost object; COSP for planned; join AUFK for order description; CSKS for cost center. Select OBJNR or order, cost element, SUM(amount).
**Top N (top 20 customers, top 10 materials, top 20 vendors):** Select the dimension (customer, material, vendor), SUM of amount/revenue; group by that dimension; order by the sum DESC; limit N (e.g. 20 or 10). Use VBRK/VBRP/KNA1 for customers, VBRP/MAKT for materials, EKPO/EKKO/LFA1 for vendors.
**Revenue by customer and year:** Group by KNA1.KUNNR (or NAME1), VBRK.GJAHR (or year from FKDAT); select SUM(VBRP.NETWR). Include customer name and year.
**Revenue by industry:** Join KNA1.BRSCH = T016T.BRSCH; select T016T text (brtxt) as industry, SUM(VBRP.NETWR); group by industry.
**Revenue by country:** Use KNA1.LAND1 or VBRK.LAND1; group by country; SUM(VBRP.NETWR).
**Revenue by customer group (KNVV):** Join VBRK.KUNAG = KNVV.KUNNR (and KNA1); group by KNVV.KDGRP; SUM(VBRP.NETWR).
**Revenue by sales org / distribution channel:** Group by VBRK.VKORG, VBRK.VTWEG; select vkorg, vtweg, SUM(VBRP.NETWR).
**Incoterms:** Select VBRK.INCO1, INCO2, SUM(VBRP.NETWR); group by inco1, inco2.
**Purchasing – total quantity and cost by material:** EKPO: SUM(MENGE) as total_quantity, SUM(NETWR) or SUM(quantity * netpr) as total_cost; group by MATNR; add MAKT for material name. For "by material and plant" group by MATNR, WERKS.
**Purchasing – average unit price (netpr):** EKPO: MATNR, AVG(NETPR) as avg_price; group by MATNR; join MAKT for name.
**AR / open balance / aging:** BSAD or BSEG with KNA1; group by customer; SUM(DMBTR or amount). For aging use clearing date (AUGDT) buckets.
**Vendor spend / invoice totals:** RBKP, RSEG or EKPO, EKKO; join LFA1 on LIFNR; group by vendor (LIFNR or name); SUM(amount). For "by vendor and year" add EKKO.BEDAT or RBKP year.
**Standard cost (KEKO, CKMLCR):** KEKO/KEPH/CKIS for cost breakdown; CKMLCR for stprs, salk3 by material. Join MARA, MAKT for material name.
**BOM (STKO, STPO, MAST):** MAST links material to BOM (STLNR); STKO header, STPO has IDNRK (component), MENGE; join STPO.IDNRK to MARA/MAKT for component name.
**FAGLFLEXA segment / rcntr / rfarea / pprctr:** Use FAGLFLEXA columns segment, rcntr (cost center), rfarea (functional area), pprctr (partner profit center) when question asks for these dimensions; group by prctr and the requested dimension.
**CKMLPP (variances):** Join CKMLPP to CKMLCR/MARA/MAKT; select material, variance fields (planned vs actual); group by material when question asks materials with large variances.
**MVKE (material hierarchy):** Join MVKE to VBRP on MATNR (and VBELN/VKORG/VTWEG if needed); use PRODH for product hierarchy; group by PRODH or region; SUM(VBRP.NETWR).
**KNVP (partner roles):** Use KNVP (PARVW = payer, bill-to, ship-to); join KNA1 on KUNNR; select partner function, customer, name.
**VBAP/VBEP (open quantity, backorder):** Use VBAP (open quantity), VBEP (schedule); join to LIPS for delivered; compare open vs delivered by material or order.
**Write-offs / credit notes:** Use BSEG or BSAD with BSHKZ or similar; group by customer and year; SUM(amount) for write-offs.
**Materials in VBRP but not in EKPO (or vice versa):** Select MATNR from VBRP EXCEPT (or LEFT JOIN EKPO WHERE EKPO.MATNR IS NULL) for "in sales not in purchasing"; reverse for "in purchasing never sold". Use MARA/MAKT for material name.
**Cost of jacket/Harley (purchase cost):** EKPO + MAKT: filter MAKT.MAKTX ILIKE '%jacket%' or '%Harley%'; SUM(NETWR) or SUM(MENGE*NETPR) by material/plant; join EKPO.MATNR = MAKT.MATNR, MAKT.SPRAS = 'E'.
**Cost of jacket-related postings by profit center:** Use FAGLFLEXA: group by prctr (profit center); SUM(hsl) as total_cost. If MAKT or VBRP is in tables and schema allows linking material to FAGLFLEXA (e.g. via cost element or segment), add filter or join for jacket materials; otherwise return cost by profit center only. Use description "profit_center" and "total_cost".
**Purchase order totals by material:** EKPO: group by MATNR; select MATNR, SUM(MENGE) as total_quantity, SUM(NETWR) or SUM(MENGE*NETPR) as total_value; join MAKT for MAKTX; order by total_value or total_quantity DESC; limit 100.
**Compare sales data with invoice data:** Select billing side (VBRK/VBRP: document, customer, SUM(NETWR)) and invoice side (RBKP/RSEG: document, vendor, amount). If same scope, join or union; otherwise return two logical sets (e.g. total sales from VBRP vs total invoice amount from RSEG by period/customer). Use KNA1 for customer name when comparing by customer.
**Compare 2023 vs 2024 sales (or two specific years):** Filter VBRK.GJAHR in (2023, 2024) or EXTRACT(YEAR FROM VBRK.FKDAT) in (2023, 2024). Group by year; select year, SUM(VBRP.NETWR) as total_sales. Optionally include customer or product for breakdown.
**Revenue last 30 days:** Filter VBRK.FKDAT >= CURRENT_DATE - INTERVAL '30 days'. Select billing doc, customer, date, amount from VBRK, VBRP, KNA1. Order by FKDAT DESC.
**Sales for 2024:** Filter VBRK.GJAHR = 2024 or EXTRACT(YEAR FROM VBRK.FKDAT) = 2024. Select as needed; SUM(NETWR) for total.
**Vague or short questions:** If the user asks something generic (e.g. "show me sales", "revenue", "what did we sell"), produce a reasonable default: e.g. billing docs with customer, date, amount; order by amount or date DESC; limit 100. Never return empty columns.
**All columns and filters must use only column names from the mappings above.**

Return JSON only:
{{
  "tables": [{{ "name": "...", "description": "..." }}],
  "columns": [{{ "table": "...", "name": "...", "description": "...", "agg": null or "SUM"/"AVG"/"COUNT"/"MIN"/"MAX" }}],
  "joins": [{{ "left": "...", "right": "...", "on": "...", "type": "inner" or "left" }}],
  "filters": [{{ "lhs": "...", "operator": "...", "rhs": "..." }}],
  "order_by": [{{ "table": "...", "column": "...", "direction": "DESC" or "ASC" }}],
  "group_by": [{{ "table": "...", "column": "..." }}],
  "limit": 100
}}
"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = (resp.choices[0].message.content or "").strip()
            spec = _safe_json_extract_adaptive(content)
            if spec and (spec.get("columns") or spec.get("tables")):
                return spec
        except Exception as e:
            logger.warning("_generate_sql_json_adaptive attempt %d failed: %s", attempt + 1, e)
    return {}


def _is_last_best_sales_query(question: str) -> bool:
    """True if the question is asking for last/best/recent sales (we handle these with a direct minimal spec)."""
    if not (question or "").strip():
        return False
    q = (question or "").lower()
    # Dynamic: sales, revenue, billing, sold, invoices + recency/top/best
    sales_related = any(x in q for x in ("sales", "revenue", "billing", "sold", "invoices", "billed"))
    recency_or_top = any(x in q for x in (
        "last ", "best ", "recent ", "latest ", "top ", "show me ", "show ", "what are the ",
        "how much did we ", "recent sales", "last sales", "best sales", "top sales"
    ))
    return (sales_related and recency_or_top) or (
        any(x in q for x in (
            "last sales", "best sales", "recent sales", "latest sales", "show me sales",
            "show sales", "last best sales", "top sales", "recent sales", "what are our sales"
        ))
        or (("last" in q or "best" in q or "recent" in q or "latest" in q) and "sales" in q)
    )


def _is_link_faglflexa_customers_products_query(question: str) -> bool:
    """True if the question asks to link FAGLFLEXA profit center costs to customers and/or products."""
    if not (question or "").strip():
        return False
    q = (question or "").lower()
    # Dynamic: accept "link", "profit center costs back to", "attribute costs to", "cost by profit center and customer"
    has_fagl = "faglflexa" in q or ("profit center" in q and "cost" in q)
    has_link_intent = any(x in q for x in ("link", "back to", "attribute", "connect", "tie "))
    has_customer_product = any(x in q for x in ("customer", "product", "major", "client"))
    return has_fagl and (has_link_intent or ("profit center" in q and has_customer_product))


def _is_cost_by_profit_center_query(question: str) -> bool:
    """True if the question asks for cost (or balance) by profit center — use FAGLFLEXA only."""
    if not (question or "").strip():
        return False
    q = (question or "").lower()
    has_pc = any(x in q for x in ("profit center", "profit centre", "prctr", "profitcenter"))
    if not has_pc:
        return False
    # Dynamic: any cost/balance/amount + profit center, or explicit phrases
    has_cost_balance = any(x in q for x in ("cost", "costs", "balance", "amount", "total ", "breakdown", "trend", "compare", "gl account"))
    return (
        has_cost_balance
        or any(
            x in q for x in (
                "cost by profit center", "total cost by profit center", "cost by profit centre",
                "costs per profit center", "balance by profit center", "compare cost",
                "costs between", "two profit center", "monthly cost trend", "cost trend",
                "profit center and gl account", "cost by profit center and gl", "by profit center"
            )
        )
    )


def _is_internal_order_question(question: str) -> bool:
    """True if the question is about internal orders (AUFK)."""
    if not (question or "").strip():
        return False
    q = (question or "").lower()
    return any(x in q for x in (
        "internal order", "internal orders", "aufk", "order master",
        "orders by profit center", "orders by cost center", "orders by company code",
        "project order", "maintenance order", "cost by internal order",
    ))


def _is_purchase_order_question(question: str) -> bool:
    """True if the question is about purchase orders / PO totals / vendor spend (for schema-driven table fallback)."""
    if not (question or "").strip():
        return False
    q = (question or "").lower()
    return any(x in q for x in (
        "purchase order", "order totals", "po totals", "purchase order totals",
        "vendor spend", "purchased", "procurement", "purchase totals", "by material",
    )) or ("purchase" in q and ("total" in q or "by material" in q or "totals" in q))


def _is_ekpo_purchasing_query(question: str) -> bool:
    """True if the question asks for purchased quantity/cost by material, plant, or vendor using EKPO."""
    if not (question or "").strip():
        return False
    q = (question or "").lower()
    purchase_related = any(x in q for x in (
        "ekpo", "purchase", "purchased", "buy", "bought", "procure", "procurement", "po ",
        "purchase order", "order totals", "po totals", "vendor spend"
    ))
    dimension_or_value = any(x in q for x in (
        "quantity", "quantities", "cost", "costs", "material", "materials", "plant", "werks",
        "netpr", "top 10", "top 20", "jacket", "harley", "for each material", "by material",
        "by plant", "average", "unit price", "total ", "totals", "list products", "list materials",
        "cheap", "high volume", "spend by", "vendor"
    ))
    return purchase_related and (dimension_or_value or "totals" in q or "by material" in q)


def _build_minimal_last_sales_spec(
    question: str,
    selected_tables: List[str],
    column_mappings: Dict[str, Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """
    Build a minimal valid spec for "last/best/recent sales" when the LLM returns empty.
    Uses VBRK, VBRP, KNA1 with standard columns so a query can be generated.
    """
    q_lower = (question or "").lower()
    include_year = "year" in q_lower or "including year" in q_lower
    # Resolve selected_tables to actual keys in column_mappings (case-insensitive)
    mapping_lower = {k.lower(): k for k in column_mappings.keys()}
    tables_in_mapping = []
    for t in selected_tables:
        actual = mapping_lower.get((t or "").lower())
        if actual:
            tables_in_mapping.append(actual)

    if not tables_in_mapping:
        return None

    # Prefer VBRP, VBRK, KNA1 order for building joins/columns
    vbrp = next((t for t in tables_in_mapping if t.upper() == "VBRP"), None)
    vbrk = next((t for t in tables_in_mapping if t.upper() == "VBRK"), None)
    kna1 = next((t for t in tables_in_mapping if t.upper() == "KNA1"), None)
    if not vbrp or not vbrk:
        return None

    def has_col(tbl: str, col: str) -> bool:
        cols = column_mappings.get(tbl, {})
        return col.lower() in {c.lower() for c in cols.keys()}

    def col_name(tbl: str, col: str) -> str:
        cols = column_mappings.get(tbl, {})
        for k, v in cols.items():
            if k.lower() == col.lower():
                return k
        return col

    columns: List[Dict[str, Any]] = []
    if vbrp and has_col(vbrp, "netwr"):
        columns.append({"table": vbrp, "name": col_name(vbrp, "netwr"), "description": "amount", "agg": None})
    if vbrk and has_col(vbrk, "vbeln"):
        columns.append({"table": vbrk, "name": col_name(vbrk, "vbeln"), "description": "billing_doc", "agg": None})
    if vbrk and has_col(vbrk, "fkdat"):
        columns.append({"table": vbrk, "name": col_name(vbrk, "fkdat"), "description": "billing_date", "agg": None})
    if include_year and vbrk and has_col(vbrk, "gjahr"):
        columns.append({"table": vbrk, "name": col_name(vbrk, "gjahr"), "description": "year", "agg": None})
    if vbrk and has_col(vbrk, "waerk"):
        columns.append({"table": vbrk, "name": col_name(vbrk, "waerk"), "description": "currency", "agg": None})
    if kna1 and has_col(kna1, "name1"):
        columns.append({"table": kna1, "name": col_name(kna1, "name1"), "description": "customer_name", "agg": None})
    if kna1 and has_col(kna1, "kunnr"):
        columns.append({"table": kna1, "name": col_name(kna1, "kunnr"), "description": "customer_number", "agg": None})

    if not columns:
        return None

    joins: List[Dict[str, Any]] = []
    if vbrp and vbrk and has_col(vbrp, "vbeln") and has_col(vbrk, "vbeln"):
        joins.append({"left": vbrp, "right": vbrk, "on": f"{vbrp}.vbeln = {vbrk}.vbeln", "type": "left"})
    if vbrk and kna1 and has_col(vbrk, "kunag") and has_col(kna1, "kunnr"):
        joins.append({"left": vbrk, "right": kna1, "on": f"{vbrk}.kunag = {kna1}.kunnr", "type": "left"})

    order_col = "netwr" if has_col(vbrp, "netwr") else "fkdat" if vbrk and has_col(vbrk, "fkdat") else None
    order_table = vbrp if (order_col == "netwr") else vbrk
    order_by = []
    if order_col and has_col(order_table, order_col):
        order_by.append({"table": order_table, "column": col_name(order_table, order_col), "direction": "DESC"})

    spec: Dict[str, Any] = {
        "tables": [{"name": t, "description": t} for t in tables_in_mapping],
        "columns": columns,
        "joins": joins,
        "filters": [],
        "order_by": order_by,
        "group_by": [],
        "limit": 100,
    }
    return spec


def _build_minimal_ekpo_spec(
    question: str,
    selected_tables: List[str],
    column_mappings: Dict[str, Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """
    Build a minimal valid spec for EKPO purchasing (quantity/cost by material, plant; jacket filter).
    Uses EKPO, EKKO, MAKT with MATNR, MENGE, NETPR, WERKS so a query can be generated.
    """
    mapping_lower = {k.lower(): k for k in column_mappings.keys()}
    tables_in_mapping = []
    for t in selected_tables:
        actual = mapping_lower.get((t or "").lower())
        if actual:
            tables_in_mapping.append(actual)
    ekpo = next((t for t in tables_in_mapping if t.upper() == "EKPO"), None)
    if not ekpo:
        return None

    def has_col(tbl: str, col: str) -> bool:
        cols = column_mappings.get(tbl, {})
        return col.lower() in {c.lower() for c in cols.keys()}

    def col_name(tbl: str, col: str) -> str:
        cols = column_mappings.get(tbl, {})
        for k, v in cols.items():
            if k.lower() == col.lower():
                return k
        return col

    makt = next((t for t in tables_in_mapping if t.upper() == "MAKT"), None)
    ekko = next((t for t in tables_in_mapping if t.upper() == "EKKO"), None)
    q_lower = (question or "").lower()
    by_plant = "plant" in q_lower or "werks" in q_lower
    avg_price = "average" in q_lower or "avg" in q_lower or "netpr" in q_lower and "average" in q_lower

    columns: List[Dict[str, Any]] = []
    group_by: List[Dict[str, str]] = []
    if has_col(ekpo, "matnr"):
        columns.append({"table": ekpo, "name": col_name(ekpo, "matnr"), "description": "material", "agg": None})
        group_by.append({"table": ekpo, "column": col_name(ekpo, "matnr")})
    if by_plant and has_col(ekpo, "werks"):
        columns.append({"table": ekpo, "name": col_name(ekpo, "werks"), "description": "plant", "agg": None})
        group_by.append({"table": ekpo, "column": col_name(ekpo, "werks")})
    if has_col(ekpo, "menge"):
        columns.append({"table": ekpo, "name": col_name(ekpo, "menge"), "description": "total_quantity", "agg": "SUM"})
    if has_col(ekpo, "netpr"):
        if avg_price:
            columns.append({"table": ekpo, "name": col_name(ekpo, "netpr"), "description": "avg_unit_price", "agg": "AVG"})
        else:
            columns.append({"table": ekpo, "name": col_name(ekpo, "netpr"), "description": "total_cost", "agg": "SUM"})
    if makt and has_col(makt, "maktx"):
        columns.append({"table": makt, "name": col_name(makt, "maktx"), "description": "material_description", "agg": None})
        if has_col(ekpo, "matnr") and has_col(makt, "matnr"):
            group_by.append({"table": makt, "column": col_name(makt, "matnr")})

    if not columns:
        return None

    joins: List[Dict[str, Any]] = []
    if ekko and has_col(ekpo, "ebeln") and has_col(ekko, "ebeln"):
        joins.append({"left": ekpo, "right": ekko, "on": f"{ekpo}.ebeln = {ekko}.ebeln", "type": "left"})
    if makt and has_col(ekpo, "matnr") and has_col(makt, "matnr"):
        joins.append({"left": ekpo, "right": makt, "on": f"{ekpo}.matnr = {makt}.matnr", "type": "left"})

    order_by = []
    if has_col(ekpo, "netpr"):
        order_by.append({"table": ekpo, "column": "total_cost" if not avg_price else "avg_unit_price", "direction": "DESC"})
    elif has_col(ekpo, "menge"):
        order_by.append({"table": ekpo, "column": "total_quantity", "direction": "DESC"})

    spec: Dict[str, Any] = {
        "tables": [{"name": t, "description": t} for t in tables_in_mapping],
        "columns": columns,
        "joins": joins,
        "filters": [],
        "order_by": order_by,
        "group_by": group_by,
        "limit": 100,
    }
    return spec


def _resolve_faglflexa_table_and_mappings(db: Session) -> Tuple[Optional[str], Dict[str, Dict[str, str]]]:
    """
    Resolve FAGLFLEXA table name and column mappings. Tries introspection with 'FAGLFLEXA',
    then case-insensitive lookup in get_table_names(), then db_table_mapping.json.
    Returns (actual_table_name, column_mappings). Either can be empty if not found.
    """
    column_mappings = _introspect_columns(db, ["FAGLFLEXA"])
    if column_mappings and any(t.upper() == "FAGLFLEXA" for t in column_mappings.keys()):
        return next((t for t in column_mappings.keys() if t.upper() == "FAGLFLEXA"), None), column_mappings
    insp = inspect(db.bind)
    all_tables = insp.get_table_names()
    actual_name = next((t for t in all_tables if t.lower() == "faglflexa"), None)
    if actual_name:
        column_mappings = _introspect_columns(db, [actual_name])
        if column_mappings:
            return actual_name, column_mappings
    try:
        root = Path(__file__).resolve().parent.parent
        path = root / "db_table_mapping.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            entry = raw.get("FAGLFLEXA") or raw.get("faglflexa")
            if isinstance(entry, dict) and isinstance(entry.get("columns"), dict):
                cols = {k: str(v) for k, v in entry["columns"].items()}
                if cols:
                    table_name = actual_name or "FAGLFLEXA"
                    return table_name, {table_name: cols}
    except Exception as e:
        logger.warning("_resolve_faglflexa_table_and_mappings: could not load mapping file: %s", e)
    return actual_name or None, column_mappings or {}


def _run_faglflexa_cost_by_profit_center_sql(
    db: Session,
    table_name: str,
    last_24_months: bool = False,
    column_mappings: Optional[Dict[str, Dict[str, str]]] = None,
) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    """
    Run a minimal 'cost by profit center' (and optionally GL account / last 24 months) query.
    Uses actual column names from column_mappings when provided (for correct DB casing).
    Tries table_name then lowercase (PostgreSQL often has lowercase tables).
    """
    cols = (column_mappings or {}).get(table_name) or {}
    def _name(logical: str) -> str:
        for k in cols.keys():
            if k.lower() == logical.lower():
                return k
        return logical

    def _build_and_run(tname: str) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
        try:
            prctr, racct, hsl, ryear, poper, rtcur = _name("prctr"), _name("racct"), _name("hsl"), _name("ryear"), _name("poper"), _name("rtcur")
            if last_24_months:
                sql = (
                    f'SELECT "{tname}"."{prctr}" AS profit_center, "{tname}"."{racct}" AS gl_account, '
                    f'"{tname}"."{ryear}" AS fiscal_year, "{tname}"."{poper}" AS posting_period, '
                    f'SUM("{tname}"."{hsl}") AS total_cost, "{tname}"."{rtcur}" AS currency '
                    f'FROM "{tname}" WHERE "{tname}"."{prctr}" IS NOT NULL '
                    f'AND "{tname}"."{ryear}" IN (EXTRACT(YEAR FROM CURRENT_DATE)::text, (EXTRACT(YEAR FROM CURRENT_DATE) - 1)::text) '
                    f'GROUP BY "{tname}"."{prctr}", "{tname}"."{racct}", "{tname}"."{ryear}", "{tname}"."{poper}", "{tname}"."{rtcur}" '
                    f'ORDER BY total_cost DESC NULLS LAST LIMIT 200'
                )
            else:
                sql = (
                    f'SELECT "{tname}"."{prctr}" AS profit_center, "{tname}"."{racct}" AS gl_account, '
                    f'SUM("{tname}"."{hsl}") AS total_cost, "{tname}"."{ryear}" AS fiscal_year, "{tname}"."{rtcur}" AS currency '
                    f'FROM "{tname}" WHERE "{tname}"."{prctr}" IS NOT NULL '
                    f'GROUP BY "{tname}"."{prctr}", "{tname}"."{racct}", "{tname}"."{ryear}", "{tname}"."{rtcur}" '
                    f'ORDER BY total_cost DESC NULLS LAST LIMIT 200'
                )
            rows = _run_sql(db, sql)
            return (sql, rows) if rows is not None else None
        except Exception as e:
            logger.debug("_run_faglflexa_cost_by_profit_center_sql with %r: %s", tname, e)
            return None

    result = _build_and_run(table_name)
    if result:
        return result
    if table_name != table_name.lower():
        result = _build_and_run(table_name.lower())
        if result:
            return result
    return None


def _build_minimal_faglflexa_spec(
    question: str,
    selected_tables: List[str],
    column_mappings: Dict[str, Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """
    Build a minimal valid spec for FAGLFLEXA profit center cost questions when the LLM returns empty.
    Returns FAGLFLEXA-only: profit center, GL account, SUM(hsl). Single-table spec to avoid heuristic joins.
    Finds the profit-center table by name (FAGLFLEXA) or by column signature (prctr + hsl).
    """
    mapping_lower = {k.lower(): k for k in column_mappings.keys()}

    def _has_col(tbl: str, col: str) -> bool:
        cols = column_mappings.get(tbl, {})
        return col.lower() in {c.lower() for c in cols.keys()}

    # Prefer table named FAGLFLEXA; else any table with prctr and hsl (profit center cost signature)
    fagl = None
    for t in selected_tables:
        actual = mapping_lower.get((t or "").lower())
        if actual and actual.upper() == "FAGLFLEXA":
            fagl = actual
            break
    if not fagl:
        for tbl in column_mappings:
            if _has_col(tbl, "prctr") and (_has_col(tbl, "hsl") or _has_col(tbl, "amount")):
                fagl = tbl
                break
    if not fagl:
        return None

    def has_col(tbl: str, col: str) -> bool:
        cols = column_mappings.get(tbl, {})
        return col.lower() in {c.lower() for c in cols.keys()}

    def col_name(tbl: str, col: str) -> str:
        cols = column_mappings.get(tbl, {})
        for k in cols.keys():
            if k.lower() == col.lower():
                return k
        return col

    columns: List[Dict[str, Any]] = []
    if has_col(fagl, "prctr"):
        columns.append({"table": fagl, "name": col_name(fagl, "prctr"), "description": "profit_center", "agg": None})
    if has_col(fagl, "racct"):
        columns.append({"table": fagl, "name": col_name(fagl, "racct"), "description": "gl_account", "agg": None})
    elif has_col(fagl, "cost_elem"):
        columns.append({"table": fagl, "name": col_name(fagl, "cost_elem"), "description": "cost_element", "agg": None})
    if has_col(fagl, "hsl"):
        columns.append({"table": fagl, "name": col_name(fagl, "hsl"), "description": "total_cost", "agg": "SUM"})
    if has_col(fagl, "ryear"):
        columns.append({"table": fagl, "name": col_name(fagl, "ryear"), "description": "fiscal_year", "agg": None})
    if has_col(fagl, "poper"):
        columns.append({"table": fagl, "name": col_name(fagl, "poper"), "description": "posting_period", "agg": None})
    if has_col(fagl, "rtcur"):
        columns.append({"table": fagl, "name": col_name(fagl, "rtcur"), "description": "currency", "agg": None})

    if not columns:
        return None

    joins: List[Dict[str, Any]] = []
    # Single-table spec: FAGLFLEXA only so SQL is SELECT ... FROM FAGLFLEXA GROUP BY ... (no heuristic joins)
    spec: Dict[str, Any] = {
        "tables": [{"name": fagl, "description": fagl}],
        "columns": columns,
        "joins": [],
        "filters": [],
        "order_by": [{"table": fagl, "column": "total_cost", "direction": "DESC"}] if has_col(fagl, "hsl") else [],
        "group_by": [{"table": fagl, "column": col_name(fagl, "prctr")}] if has_col(fagl, "prctr") else [],
        "limit": 100,
    }
    if has_col(fagl, "racct"):
        spec["group_by"].append({"table": fagl, "column": col_name(fagl, "racct")})
    elif has_col(fagl, "cost_elem"):
        spec["group_by"].append({"table": fagl, "column": col_name(fagl, "cost_elem")})
    return spec


def run_adaptive_sap_sql_agent(
    question: str,
    db: Session,
    knowledge_context: Optional[str] = None,
    time_scope: str = "current",
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
) -> Optional[SqlAgentResult]:
    """
    Run the invoice-bot-style adaptive SQL path: richer table selection and SQL spec
    generation so each query gets question-specific SQL. Uses Postgres execution.
    Returns SqlAgentResult on success, None on failure (caller should fall back to run_sap_sql_agent).
    """
    client = _get_openai_client()
    if not client:
        return None

    try:
        # Direct path for "Link FAGLFLEXA profit center costs back to customers/products" — ensure FAGLFLEXA + sales tables
        if _is_link_faglflexa_customers_products_query(question):
            selected_tables = ["FAGLFLEXA", "VBRK", "VBRP", "KNA1", "MAKT"]
            logger.info("run_adaptive_sap_sql_agent: using direct FAGLFLEXA-link path, tables=%s", selected_tables)
            column_mappings = _introspect_columns(db, selected_tables)
            if not column_mappings or not any(t.upper() == "FAGLFLEXA" for t in column_mappings.keys()):
                fagl_table, fagl_mappings = _resolve_faglflexa_table_and_mappings(db)
                if fagl_table and fagl_mappings:
                    selected_tables = list(fagl_mappings.keys())
                    column_mappings = fagl_mappings
                elif not column_mappings:
                    selected_tables = ["FAGLFLEXA"]
                    column_mappings = _introspect_columns(db, selected_tables)
            if column_mappings and (any(t.upper() == "FAGLFLEXA" for t in column_mappings.keys())):
                table_descriptions = _get_table_descriptions(db)
                spec = _generate_sql_json_adaptive(
                    question,
                    list(column_mappings.keys()),
                    column_mappings,
                    client,
                    time_scope=time_scope,
                    few_shot_examples=few_shot_examples,
                    table_descriptions=table_descriptions,
                )
                if not spec:
                    spec = _build_minimal_faglflexa_spec(question, list(column_mappings.keys()), column_mappings)
                if spec:
                    try:
                        from .invoice_bot_helpers import (
                            fix_date_filters,
                            inject_product_name_filter_if_needed,
                            inject_material_number_filter_if_needed,
                            inject_makt_single_language_if_needed,
                            ensure_delivery_chain_in_spec,
                            inject_country_filter_if_needed,
                        )
                        fix_date_filters(spec)
                        inject_product_name_filter_if_needed(question, spec)
                        inject_material_number_filter_if_needed(question, spec)
                        inject_makt_single_language_if_needed(spec)
                        ensure_delivery_chain_in_spec(spec)
                        inject_country_filter_if_needed(question, spec)
                    except Exception as e:
                        logger.warning("invoice_bot_helpers spec post-processing failed: %s", e)
                    _ensure_having_for_aggregates(spec, question)
                    _auto_enrich_spec(spec, question)
                    is_valid, validation_errors = validate_sql_spec(spec)
                    if is_valid:
                        sql = _json_to_sql_postgres(spec, column_mappings)
                        if sql:
                            rows = _run_sql(db, sql)
                            if rows:
                                logger.info("run_adaptive_sap_sql_agent: direct FAGLFLEXA-link returned %d rows", len(rows))
                                return SqlAgentResult(sql=sql, rows=rows)
                    else:
                        logger.warning("run_adaptive_sap_sql_agent: direct FAGLFLEXA-link spec validation failed: %s", validation_errors)
            # Raw SQL fallback: return cost by profit center when link spec fails
            link_table_name, link_mappings = _resolve_faglflexa_table_and_mappings(db)
            if link_table_name:
                result = _run_faglflexa_cost_by_profit_center_sql(db, link_table_name, last_24_months=False, column_mappings=link_mappings)
                if result:
                    sql, rows = result
                    logger.info("run_adaptive_sap_sql_agent: FAGLFLEXA-link raw SQL fallback returned %d rows", len(rows or []))
                    return SqlAgentResult(sql=sql, rows=rows or [])
            # Fall through to normal path if direct path didn't return
        # Direct path for "Total cost by profit center" / "cost by profit center" — FAGLFLEXA only.
        # Skip when question mentions a product (jacket, harley, etc.) so normal path + intent enrichment gets FAGLFLEXA+MAKT.
        elif _is_cost_by_profit_center_query(question) and not any(
            w in (question or "").lower() for w in ("jacket", "harley", "product", "material", "related", "posting")
        ):
            q_lower = (question or "").lower()
            last_24 = "last 24" in q_lower or "24 months" in q_lower
            logger.info("run_adaptive_sap_sql_agent: using direct cost-by-profit-center path (last_24=%s)", last_24)
            table_name, column_mappings = _resolve_faglflexa_table_and_mappings(db)
            if table_name and column_mappings:
                selected_tables = list(column_mappings.keys())
                spec = _build_minimal_faglflexa_spec(question, selected_tables, column_mappings)
                if spec:
                    try:
                        from .invoice_bot_helpers import (
                            fix_date_filters,
                            inject_product_name_filter_if_needed,
                            inject_material_number_filter_if_needed,
                            inject_makt_single_language_if_needed,
                            ensure_delivery_chain_in_spec,
                            inject_country_filter_if_needed,
                        )
                        fix_date_filters(spec)
                        inject_product_name_filter_if_needed(question, spec)
                        inject_material_number_filter_if_needed(question, spec)
                        inject_makt_single_language_if_needed(spec)
                        ensure_delivery_chain_in_spec(spec)
                        inject_country_filter_if_needed(question, spec)
                    except Exception as e:
                        logger.warning("invoice_bot_helpers spec post-processing failed: %s", e)
                    _ensure_having_for_aggregates(spec, question)
                    _auto_enrich_spec(spec, question)
                    is_valid, validation_errors = validate_sql_spec(spec)
                    if is_valid:
                        sql = _json_to_sql_postgres(spec, column_mappings)
                        if sql:
                            rows = _run_sql(db, sql)
                            if rows:
                                logger.info("run_adaptive_sap_sql_agent: direct cost-by-profit-center returned %d rows", len(rows))
                                return SqlAgentResult(sql=sql, rows=rows)
                    else:
                        logger.warning("run_adaptive_sap_sql_agent: direct cost-by-profit-center spec validation failed: %s", validation_errors)
            # Raw SQL fallback when spec path fails or we have table from mapping file
            if table_name:
                result = _run_faglflexa_cost_by_profit_center_sql(db, table_name, last_24_months=last_24, column_mappings=column_mappings)
                if result:
                    sql, rows = result
                    logger.info("run_adaptive_sap_sql_agent: FAGLFLEXA raw SQL fallback returned %d rows", len(rows or []))
                    return SqlAgentResult(sql=sql, rows=rows or [])
            # Fall through to normal path if FAGLFLEXA not in DB or all attempts failed
        # Direct path for "last/best/recent sales" — skip LLM, use fixed tables + minimal spec so these always work
        elif _is_last_best_sales_query(question):
            selected_tables = ["VBRK", "VBRP", "KNA1"]
            logger.info("run_adaptive_sap_sql_agent: using direct last-sales path, tables=%s", selected_tables)
            column_mappings = _introspect_columns(db, selected_tables)
            if not column_mappings:
                logger.warning("run_adaptive_sap_sql_agent: no column mappings for direct last-sales path")
            else:
                spec = _build_minimal_last_sales_spec(question, selected_tables, column_mappings)
                if spec:
                    # Run the same post-process, validate, build SQL, execute as below
                    try:
                        from .invoice_bot_helpers import (
                            fix_date_filters,
                            inject_product_name_filter_if_needed,
                            inject_material_number_filter_if_needed,
                            inject_makt_single_language_if_needed,
                            ensure_delivery_chain_in_spec,
                            inject_country_filter_if_needed,
                        )
                        fix_date_filters(spec)
                        inject_product_name_filter_if_needed(question, spec)
                        inject_material_number_filter_if_needed(question, spec)
                        inject_makt_single_language_if_needed(spec)
                        ensure_delivery_chain_in_spec(spec)
                        inject_country_filter_if_needed(question, spec)
                    except Exception as e:
                        logger.warning("invoice_bot_helpers spec post-processing failed: %s", e)
                    _ensure_having_for_aggregates(spec, question)
                    _auto_enrich_spec(spec, question)
                    is_valid, validation_errors = validate_sql_spec(spec)
                    if is_valid:
                        sql = _json_to_sql_postgres(spec, column_mappings)
                        if sql:
                            logger.info("run_adaptive_sap_sql_agent: direct last-sales SQL (first 300 chars): %s", (sql or "")[:300])
                            rows = _run_sql(db, sql)
                            if rows:
                                return SqlAgentResult(sql=sql, rows=rows)
                            else:
                                logger.warning("run_adaptive_sap_sql_agent: direct last-sales spec validation failed: %s", validation_errors)
            # If direct path didn't return, fall through to LLM path
        # Direct path for EKPO purchasing (total quantity/cost by material, jacket, top 10 products)
        elif _is_ekpo_purchasing_query(question):
            selected_tables = ["EKKO", "EKPO", "MAKT", "MARA"]
            logger.info("run_adaptive_sap_sql_agent: using direct EKPO purchasing path, tables=%s", selected_tables)
            column_mappings = _introspect_columns(db, selected_tables)
            if column_mappings:
                spec = _build_minimal_ekpo_spec(question, selected_tables, column_mappings)
                if not spec:
                    table_descriptions = _get_table_descriptions(db)
                    spec = _generate_sql_json_adaptive(
                        question, list(column_mappings.keys()), column_mappings, client,
                        time_scope=time_scope, few_shot_examples=few_shot_examples,
                        table_descriptions=table_descriptions,
                    )
                if spec:
                    try:
                        from .invoice_bot_helpers import (
                            fix_date_filters,
                            inject_product_name_filter_if_needed,
                            inject_material_number_filter_if_needed,
                            inject_makt_single_language_if_needed,
                            ensure_delivery_chain_in_spec,
                            inject_country_filter_if_needed,
                        )
                        fix_date_filters(spec)
                        inject_product_name_filter_if_needed(question, spec)
                        inject_material_number_filter_if_needed(question, spec)
                        inject_makt_single_language_if_needed(spec)
                        ensure_delivery_chain_in_spec(spec)
                        inject_country_filter_if_needed(question, spec)
                    except Exception as e:
                        logger.warning("invoice_bot_helpers spec post-processing failed: %s", e)
                    _ensure_having_for_aggregates(spec, question)
                    _auto_enrich_spec(spec, question)
                    is_valid, validation_errors = validate_sql_spec(spec)
                    if is_valid:
                        sql = _json_to_sql_postgres(spec, column_mappings)
                        if sql:
                            rows = _run_sql(db, sql)
                            if rows:
                                logger.info("run_adaptive_sap_sql_agent: direct EKPO path returned %d rows", len(rows))
                                return SqlAgentResult(sql=sql, rows=rows)
            # Fall through to normal path if EKPO not available or no rows

        selected_tables = _pick_tables_adaptive(question, client, db, knowledge_context)
        # Fallback when LLM returns no tables: infer from common patterns so short queries still work
        if not selected_tables:
            q_lower = (question or "").lower()
            if any(x in q_lower for x in ("link faglflexa", "faglflexa profit center", "profit center costs back to", "profit center and customer")) or (
                "faglflexa" in q_lower and ("customer" in q_lower or "product" in q_lower or "link" in q_lower)
            ):
                selected_tables = ["FAGLFLEXA", "VBRK", "VBRP", "KNA1", "MAKT"]
            elif any(x in q_lower for x in ("profit center", "gl account", "cost by profit", "cost by gl", "compare cost", "costs between", "two profit center", "top 20 profit centers", "top 10 profit centers", "list top", "profit centers by cost")):
                selected_tables = ["FAGLFLEXA"]
            elif any(x in q_lower for x in ("jacket", "harley")) and any(x in q_lower for x in ("profit center", "profit centre")) and any(x in q_lower for x in ("cost", "postings")):
                selected_tables = ["FAGLFLEXA", "MAKT", "VBRP"]
            elif any(x in q_lower for x in ("ekpo", "purchase", "purchased quantity", "purchase cost", "vendor spend", "total purchased quantity", "netpr", "matnr werks", "purchase order totals", "po totals", "order totals by material")):
                selected_tables = ["EKKO", "EKPO", "MAKT", "MARA"]
            elif any(x in q_lower for x in ("compare sales", "sales data", "invoice data") and ("invoice" in q_lower or "compare" in q_lower)):
                selected_tables = ["VBRK", "VBRP", "RBKP", "RSEG", "KNA1"]
            elif any(x in q_lower for x in ("compare", "2023", "2024", "vs")) and ("sales" in q_lower or "revenue" in q_lower):
                selected_tables = ["VBRK", "VBRP", "KNA1"]
            elif any(x in q_lower for x in ("sales for 2024", "revenue last 30", "last 30 days", "2024 sales")):
                selected_tables = ["VBRK", "VBRP", "KNA1"]
            elif any(x in q_lower for x in ("sales by country and industry", "country and industry", "which industry", "highest revenues", "industry has highest")):
                selected_tables = ["VBRK", "VBRP", "KNA1", "T016T"]
            elif any(x in q_lower for x in ("jacket", "harley") and any(x in q_lower for x in ("purchase", "ekpo", "cost", "quantity"))):
                selected_tables = ["EKKO", "EKPO", "MAKT", "MARA"]
            elif any(x in q_lower for x in ("resb", "lips", "discrepancy")) and ("reserved" in q_lower or "delivered" in q_lower or "discrepancy" in q_lower):
                selected_tables = ["RESB", "LIPS", "MARA", "MAKT", "MKPF"]
            elif any(x in q_lower for x in ("resb", "reserved")) and ("quantity" in q_lower or "highest" in q_lower or "total quantity" in q_lower):
                selected_tables = ["RESB", "MARA", "MAKT"]
            elif any(x in q_lower for x in ("mkpf", "stock movement", "reservation", "issued from inventory", "mseg", "slow-moving", "reserved quantity", "no movements", "posting dates")):
                selected_tables = ["MKPF", "MSEG", "MARA", "MAKT"]
            elif any(x in q_lower for x in ("marc", "plant master", "mrp", "configurable", "batch-managed", "marm", "on-hand", "valuation", "planning")):
                selected_tables = ["MARC", "MARA", "MAKT"]
            elif any(x in q_lower for x in ("keko", "ckmlcr", "ckis", "standard cost", "stprs", "salk3", "bom", "stko", "stpo", "mast", "component", "ckmlpp", "variance", "planned vs actual")):
                selected_tables = ["KEKO", "KEPH", "CKIS", "MARA", "MAKT"] if "ckmlpp" not in q_lower else ["CKMLCR", "CKMLPP", "MARA", "MAKT"]
            elif any(x in q_lower for x in ("ckmlcr", "costed materials", "stock value", "finished goods")):
                selected_tables = ["CKMLCR", "MARA", "MAKT"]
            elif any(x in q_lower for x in ("konv", "condition", "price condition", "discount", "pr00", "base price", "kschl", "knumv", "rebate", "ra01", "k007", "kappl", "cash discount", "list price", "net price")):
                selected_tables = ["KONV", "VBRK", "VBRP", "MAKT", "KNA1"]
            elif any(x in q_lower for x in ("bsad", "bseg", "ar ", "receivable", "aging", "write-off", "open ar", "payment terms", "zterm", "overdue", "bkpf", "write off", "credit note")):
                selected_tables = ["BSAD", "BSEG", "KNA1"]
            elif any(x in q_lower for x in ("rbkp", "rseg", "vendor invoice", "lfa1", " spend by vendor", "vendor balance", "lfb1", "invoice amount", "ap aging", "payables", "rmwwr", "matkl", "pareto", "80%", "lead time", "ekorg")):
                selected_tables = ["LFA1", "EKKO", "EKPO", "RBKP", "RSEG"]
            elif any(x in q_lower for x in ("margin", "profitability", "revenue minus cost", "improving margin", "margin year over year", "margin trend", "products with improving")):
                selected_tables = ["VBRK", "VBRP", "EKPO", "MAKT"]
            elif any(x in q_lower for x in ("deliver", "likp", "lips", "delivered quantity", "delivery document", "vstel", "vbep")):
                selected_tables = ["LIKP", "LIPS", "VBRP", "KNA1"]
            elif any(x in q_lower for x in ("aufk", "coep", "cosp", "csks", "cepc", "internal order", "cost center", "cost by order")):
                selected_tables = ["AUFK", "COEP", "CSKS"]
            elif any(x in q_lower for x in ("by year", "per year", "sales by year", "revenue by year", "revenue by customer and year", "total revenue by customer")):
                selected_tables = ["VBRK", "VBRP", "KNA1"]
            elif any(x in q_lower for x in ("industry", "t016t", "brsch", "top 10 industries")):
                selected_tables = ["VBRK", "VBRP", "KNA1", "T016T"]
            elif any(x in q_lower for x in ("customer group", "knvv", "kdgrp", "sales org", "vkorg", "vtweg", "distribution channel")):
                selected_tables = ["VBRK", "VBRP", "KNA1", "KNVV"]
            elif any(x in q_lower for x in ("incoterm", "inco1", "inco2", "revenue split")):
                selected_tables = ["VBRK", "VBRP"]
            elif any(x in q_lower for x in ("top 20 customers", "top 20 materials", "top 10", "billed revenue", "invoice value", "billed documents", "top 20 documents", "average invoice", "average selling price", "netwr", "fkimg")):
                selected_tables = ["VBRK", "VBRP", "KNA1", "MAKT"]
            elif any(x in q_lower for x in ("declining revenue", "no sales", "last 12 months", "historical revenue", "no recent")):
                selected_tables = ["VBRK", "VBRP", "KNA1"]
            elif any(x in q_lower for x in ("mix", "country", "industry")) and ("revenue" in q_lower or "sales" in q_lower):
                selected_tables = ["VBRK", "VBRP", "KNA1", "T016T"]
            elif any(x in q_lower for x in ("mvke", "prodh", "material hierarchy", "region")):
                selected_tables = ["VBRK", "VBRP", "KNA1", "MAKT"]
            elif any(x in q_lower for x in ("knvp", "ship-to", "payer", "bill-to", "partner relationship")):
                selected_tables = ["KNA1", "KNVP"]
            elif any(x in q_lower for x in ("multiple sales areas", "sales areas")):
                selected_tables = ["VBRK", "VBRP", "KNA1", "KNVV"]
            elif any(x in q_lower for x in ("cost of the jacket", "cost of jacket", "cost of harley", "cost of the harley", "know the cost", "cost history", "do you know the cost")) and any(x in q_lower for x in ("jacket", "harley", "product", "material")):
                selected_tables = ["EKPO", "EKKO", "MAKT", "MARA"]
            elif any(x in q_lower for x in ("end-to-end", "profitability by customer", "profitability by material", "cross-functional")):
                selected_tables = ["VBRK", "VBRP", "EKPO", "KNA1", "MAKT"]
            elif any(x in q_lower for x in ("materials", "in sales", "not in purchasing", "never sold", "appear in sales", "appear in purchasing")):
                selected_tables = ["VBRP", "EKPO", "MARA", "MAKT"]
            elif any(x in q_lower for x in ("compare", "two", "cost centers", "between two cost")):
                selected_tables = ["COEP", "CSKS"]
            elif any(x in q_lower for x in ("cepc", "profit center master", "link to gl")):
                selected_tables = ["CEPC", "FAGLFLEXA"]
            elif any(x in q_lower for x in ("lfb1", "payment terms")) and "vendor" in q_lower:
                selected_tables = ["LFA1", "LFB1", "RBKP", "RSEG"]
            elif any(x in q_lower for x in ("total cost by vendor", "highest spend by vendor", "spend by vendor", "cost by vendor")):
                selected_tables = ["LFA1", "EKKO", "EKPO", "MAKT"]
            elif any(x in q_lower for x in ("invoice amounts by customer", "invoice value by customer", "top vendors by invoice")):
                selected_tables = ["VBRK", "VBRP", "KNA1"] if "customer" in q_lower else ["LFA1", "RBKP", "RSEG"]
            elif any(x in q_lower for x in ("by currency", "sales by currency", "waerk", "waers", "vendor invoice totals by currency")):
                selected_tables = ["VBRK", "VBRP"] if "vendor" not in q_lower else ["LFA1", "RBKP", "RSEG"]
            elif any(x in q_lower for x in (
                "last sales", "best sales", "recent sales", "latest sales", "show me sales",
                "show sales", "last best sales", "top sales", "recent sales"
            )) or (("last" in q_lower or "best" in q_lower or "recent" in q_lower or "latest" in q_lower) and "sales" in q_lower):
                selected_tables = ["VBRK", "VBRP", "KNA1"]
            elif any(x in q_lower for x in ("jacket", "harley")) and any(x in q_lower for x in ("revenue", "sales", "customer", "cost", "margin", "invoice")):
                selected_tables = ["VBRK", "VBRP", "KNA1", "MAKT"]
            else:
                # Dynamic intent-based fallback: map intent tokens to tables so any phrasing is handled
                intents = _get_query_intent_tokens(question)
                if "profit_center" in intents or "cost_center" in intents:
                    selected_tables = ["FAGLFLEXA"] if "customer" not in intents and "product" not in intents else ["FAGLFLEXA", "VBRK", "VBRP", "KNA1", "MAKT"]
                elif "revenue" in intents or "sales" in intents or "customer" in intents:
                    selected_tables = ["VBRK", "VBRP", "KNA1"]
                    if "material" in intents or "product" in intents or "industry" in intents:
                        selected_tables = ["VBRK", "VBRP", "KNA1", "MAKT"]
                    if "industry" in intents:
                        selected_tables = ["VBRK", "VBRP", "KNA1", "T016T"]
                elif "purchase" in intents or "vendor" in intents:
                    selected_tables = ["EKKO", "EKPO", "MAKT", "MARA"]
                elif "delivery" in intents:
                    selected_tables = ["LIKP", "LIPS", "VBRP", "KNA1"]
                elif "margin" in intents or "profit" in intents:
                    selected_tables = ["VBRK", "VBRP", "EKPO", "MAKT"]
                elif "cost" in intents and ("material" in intents or "product" in intents):
                    selected_tables = ["EKPO", "EKKO", "MAKT", "MARA"]
                elif "ask_value" in intents and not selected_tables:
                    selected_tables = ["VBRK", "VBRP", "KNA1"]
                if not selected_tables:
                    try:
                        from .invoice_bot_helpers import _extract_country_iso_from_query
                        if _extract_country_iso_from_query(question):
                            selected_tables = ["VBRK", "VBRP", "KNA1"]
                        elif any(x in q_lower for x in ("sales", "revenue", "customer", "invoice", "billed", "industry")):
                            selected_tables = ["VBRK", "VBRP", "KNA1"]
                    except Exception:
                        pass
            if selected_tables:
                logger.info("run_adaptive_sap_sql_agent: using fallback tables for short query: %s", selected_tables)
        # Intent-based enrichment: ensure tables match question intent (any wording)
        selected_tables = _enrich_tables_by_intent(question, selected_tables or [])
        if selected_tables:
            logger.info("run_adaptive_sap_sql_agent: selected_tables after intent enrichment: %s", selected_tables)
        if not selected_tables:
            logger.info("run_adaptive_sap_sql_agent: no tables selected, falling back to standard path")
            return None

        logger.info("run_adaptive_sap_sql_agent: question=%r | selected_tables=%s", question[:80], selected_tables)

        column_mappings = _introspect_columns(db, selected_tables)
        if not column_mappings:
            logger.warning("run_adaptive_sap_sql_agent: no column mappings for %s", selected_tables)
            return None

        table_descriptions = _get_table_descriptions(db)
        spec = _generate_sql_json_adaptive(
            question,
            selected_tables,
            column_mappings,
            client,
            time_scope=time_scope,
            few_shot_examples=few_shot_examples,
            table_descriptions=table_descriptions,
        )
        if not spec:
            # When LLM returns empty, use minimal spec for known question types so we still answer
            q_lower = (question or "").lower()
            if any(x in q_lower for x in ("last sales", "best sales", "recent sales", "show me sales", "top sales")) or (
                ("last" in q_lower or "best" in q_lower or "recent" in q_lower) and "sales" in q_lower
            ):
                spec = _build_minimal_last_sales_spec(question, selected_tables, column_mappings)
                if spec:
                    logger.info("run_adaptive_sap_sql_agent: using minimal last-sales spec after LLM returned empty")
            if not spec and "FAGLFLEXA" in (t.upper() for t in selected_tables) and any(
                x in q_lower for x in ("profit center", "faglflexa", "link", "cost by profit", "gl account", "jacket", "postings")
            ):
                spec = _build_minimal_faglflexa_spec(question, selected_tables, column_mappings)
                if spec:
                    logger.info("run_adaptive_sap_sql_agent: using minimal FAGLFLEXA spec after LLM returned empty")
            if not spec and "EKPO" in (t.upper() for t in selected_tables) and any(
                x in q_lower for x in ("purchase", "purchased", "quantity", "cost", "material", "jacket", "harley", "vendor", "plant", "netpr", "totals", "order totals")
            ):
                spec = _build_minimal_ekpo_spec(question, selected_tables, column_mappings)
                if spec:
                    logger.info("run_adaptive_sap_sql_agent: using minimal EKPO spec after LLM returned empty")
            if not spec:
                return None

        # Invoice-bot spec post-processing: date filters, product name, material number, MAKT language, delivery chain, country
        try:
            from .invoice_bot_helpers import (
                fix_date_filters,
                inject_product_name_filter_if_needed,
                inject_material_number_filter_if_needed,
                inject_makt_single_language_if_needed,
                ensure_delivery_chain_in_spec,
                inject_country_filter_if_needed,
            )
            fix_date_filters(spec)
            inject_product_name_filter_if_needed(question, spec)
            inject_material_number_filter_if_needed(question, spec)
            inject_makt_single_language_if_needed(spec)
            ensure_delivery_chain_in_spec(spec)
            inject_country_filter_if_needed(question, spec)
        except Exception as e:
            logger.warning("invoice_bot_helpers spec post-processing failed: %s", e)

        _ensure_having_for_aggregates(spec, question)
        _auto_enrich_spec(spec, question)

        is_valid, validation_errors = validate_sql_spec(spec)
        if not is_valid and validation_errors:
            spec = refine_query_on_error(client, question, "; ".join(validation_errors), spec)
            is_valid, _ = validate_sql_spec(spec)
        if not is_valid:
            return None

        sql = _json_to_sql_postgres(spec, column_mappings)
        logger.info("run_adaptive_sap_sql_agent: generated SQL (first 400 chars): %s", (sql or "")[:400])
        rows = _run_sql(db, sql)

        if rows:
            logger.info("run_adaptive_sap_sql_agent: success, %d rows", len(rows))
            return SqlAgentResult(sql=sql, rows=rows)

        # One retry with refinement when 0 rows
        spec = refine_query_on_error(
            client,
            question,
            "Query returned no rows. Simplify joins or remove strict filters; use all periods if needed.",
            spec,
        )
        if spec:
            sql = _json_to_sql_postgres(spec, column_mappings)
            rows = _run_sql(db, sql)
            if rows:
                return SqlAgentResult(sql=sql, rows=rows)
    except Exception as e:
        logger.warning("run_adaptive_sap_sql_agent failed for %r: %s", question[:80], e)
    return None


def _generate_sql_json(
    question: str,
    selected_tables: List[str],
    column_mappings: Dict[str, Dict[str, str]],
    client: OpenAI,
    time_scope: str = "current",
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
    table_descriptions: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Equivalent of INVOICE_BOT.generate_sql_json, but simplified and Postgres-focused.

    Supports:
    - simple SELECTs
    - aggregates (SUM/AVG/COUNT/MIN/MAX) via the optional "agg" field on columns
    - GROUP BY via a dedicated "group_by" list
    """
    # Determine date filter based on time_scope
    date_filter_instruction = ""
    if time_scope == "historical":
        date_filter_instruction = """
⏳ **TIME SCOPE: HISTORICAL DATA (1994-2010)**
- MUST add date filters to ONLY include data from 1994-01-01 to 2010-12-31
- Example: {{"lhs": "VBRK.FKDAT", "operator": ">=", "rhs": "'1994-01-01'"}}, {{"lhs": "VBRK.FKDAT", "operator": "<=", "rhs": "'2010-12-31'"}}
"""
    elif time_scope == "current":
        date_filter_instruction = """
⏳ **TIME SCOPE: CURRENT PERIOD**
- Include recent data only (no strict date filter unless user specifies)
"""
    elif time_scope == "both":
        date_filter_instruction = """
⏳ **TIME SCOPE: ALL PERIODS**
- Include ALL data from 1994 to present for comparison
"""

    few_shot_block = ""
    if few_shot_examples:
        # Keep prompt lean: only a few short examples
        cleaned_examples = []
        for ex in few_shot_examples[:5]:
            uq = str(ex.get("user_query") or "")[:500]
            sql_ex = str(ex.get("sql_query") or "")[:1500]
            if uq and sql_ex:
                cleaned_examples.append({"user_query": uq, "sql_query": sql_ex})
        if cleaned_examples:
            few_shot_block = (
                "\nRecent successful question→SQL examples (use as patterns, do NOT copy literally):\n"
                f"{json.dumps(cleaned_examples, indent=2)}\n"
            )

    # Table descriptions: config/mapping-first, fallback to SAP_TABLE_DESCRIPTIONS
    tbl_desc = table_descriptions or {}
    tables_block = {tbl: tbl_desc.get(tbl) or tbl_desc.get(tbl.upper()) or SAP_TABLE_DESCRIPTIONS.get(tbl, "") for tbl in selected_tables}

    # Also expose ALL available table descriptions as a reference catalogue.
    # This lets the LLM add a table (e.g. MAKT for a LIKE filter, KNA1 for a name lookup)
    # even if it wasn't in the pre-selected set — without a second round-trip.
    all_tables_catalogue = ""
    if tbl_desc:
        extra = {k: v for k, v in tbl_desc.items() if k not in tables_block}
        if extra:
            all_tables_catalogue = (
                "\nOther available tables you MAY add if the query requires them "
                "(include them in the 'tables' list and add the necessary join):\n"
                + json.dumps(extra, indent=2)
                + "\n"
            )

    # Config-driven: column_semantic_hints + join_rules (extensible when adding new tables)
    cfg = _get_schema_config()
    col_hints = cfg.get("column_semantic_hints") or {}
    join_rules = cfg.get("join_rules") or []
    col_hints_block = ""
    if col_hints:
        col_hints_block = (
            "\nColumn semantics (use when choosing columns – from schema_ai_config.json):\n"
            + "\n".join(f"- {k}: {v}" for k, v in list(col_hints.items())[:60])  # increased: was 25
            + "\n"
        )
    join_rules_block = ""
    if join_rules:
        join_rules_block = "\nConfigured join rules (schema_ai_config.json – use these when joining):\n" + "\n".join(
            f"- {r.get('left')} + {r.get('right')}: {r.get('on', '')}" for r in join_rules[:30]  # increased: was 20
        ) + "\n"

    semantic_block = _get_semantic_prompt_block()

    prompt = f"""
User question: "{question}"

{date_filter_instruction}

{few_shot_block}
{semantic_block}Tables selected as primary (with full column details below):
{json.dumps(tables_block, indent=2)}
{all_tables_catalogue}
Column mappings (table -> column -> short description):
{json.dumps(column_mappings, indent=2)}
{col_hints_block}
Known join patterns between these tables:
{SAP_JOIN_HINTS}
{join_rules_block}

Task:
- Choose relevant columns from these tables.
- Propose joins between tables using ONLY the business keys listed above (do NOT invent other join columns).
- **IMPORTANT – Customer queries**: VBRP/vbrp does NOT have KUNNR/KUNAG. To get customer data:
  * PREFERRED (works even without full KNA1 data): use VBRK.KUNAG directly as customer identifier.
    Group by VBRK.KUNAG for "by customer" aggregations if KNA1 has incomplete data.
  * For customer NAME: join VBRK.KUNAG = KNA1.KUNNR and use KNA1.NAME1. Use LEFT JOIN (not INNER).
  * Never add IS NOT NULL filter on KNA1 — that converts LEFT JOIN to INNER JOIN and kills results.
- **IMPORTANT – Country queries**: VBRK has its own LAND1 column (country). Prefer VBRK.LAND1 directly
  instead of joining to KNA1.LAND1 — this avoids empty results when KNA1 data is incomplete.
- **T016T (industry)**: ONLY include T016T when the question explicitly asks for "industry" or "by industry".
  * T016T has ONLY columns brsch and brtxt (no VBELN, no KUNNR).
  * Join: KNA1.brsch = T016T.brsch (NOT on VBELN).
  * SELECT T016T.brtxt for industry name (not KNA1.brsch which is just a code).
  * Do NOT add T016T for questions about products, customers, or sales alone.
- **SIMILAR RULE**: For materials, use MAKT.MAKTX (description) not MATNR (code)
- **Margin/profitability**: margin = (revenue - cost) / revenue. Revenue from VBRP.NETWR. Cost from EKPO.NETWR or CKIS.wertn joined on material. For "average margin on low products" use AVG of margin per product, filter to low-margin products, group by product. If EKPO/CKIS not available, use revenue-only analysis and note that true margin needs cost data.
- **Cost of a specific product (e.g. a jacket)**: when the question is "cost of X" or "price of X":
  * PREFERRED: use KEKO + CKIS + MAKT for the STANDARD COST.
    - Filter: MAKT.MAKTX ILIKE '%jacket%'  (or whatever product)
    - Join: KEKO.MATNR = MAKT.MATNR, KEKO.KALNR = CKIS.KALNR
    - Select: MAKT.MAKTX, KEKO.matnr, SUM(CKIS.wertn) as standard_cost
    - Note: KEKO does NOT have stprs column; standard cost total is SUM(CKIS.WERTN).
    - Alternative for unit price: use CKMLCR.stprs joined via CKMLCR.kalnr = CKMLHD.kalnr, CKMLHD.matnr = MAKT.matnr
  * ALTERNATIVE (if KEKO not available or returns nothing): use EKPO + MAKT.
    - MAKT.MAKTX ILIKE '%jacket%', join EKPO.MATNR = MAKT.MATNR, SUM(EKPO.NETWR) / NULLIF(SUM(EKPO.MENGE), 0) as unit_cost.
  * LAST RESORT: use VBRP + MAKT to show the SALES PRICE as a proxy (note: this is selling price, not cost).
    - Group by MAKT.MAKTX, compute SUM(VBRP.NETWR) / NULLIF(SUM(VBRP.FKIMG), 0) as avg_sales_price.
- Add filters only if clearly needed from the question (for dates, customers, countries, industries, products, etc.).
- Return STRICT JSON with this structure:
{{
  "tables": [{{ "name": "VBRP", "description": "..." }}],
  "columns": [
    {{ "table": "T016T", "name": "brtxt", "description": "industry_name", "agg": null }},
    {{ "table": "VBRP", "name": "NETWR", "description": "total_sales", "agg": "SUM" }}
  ],
  "joins": [
    {{ "left": "VBRP", "right": "VBRK", "on": "VBRP.VBELN = VBRK.VBELN" }},
    {{ "left": "VBRK", "right": "KNA1", "on": "VBRK.KUNAG = KNA1.KUNNR" }},
    {{ "left": "KNA1", "right": "T016T", "on": "KNA1.brsch = T016T.brsch" }}
  ],
  "filters": [{{ "lhs": "VBRK.FKDAT", "operator": ">=", "rhs": "'2024-01-01'" }}],
  "group_by": [
    {{ "table": "T016T", "column": "brtxt" }}
  ],
  "having": [
    {{ "lhs": "total_sales", "operator": ">", "rhs": "0" }}
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
- CRITICAL: For "order_by", you MUST reference a column by its "description" field from the "columns" array.
  Example: If you have {{"table": "VBRP", "name": "NETWR", "description": "total_sales", "agg": "SUM"}},
  then order_by should be ["total_sales DESC"], NOT ["NETWR DESC"] or ["billing_item_net_value DESC"].
- For questions like:
    * "which industry has highest revenues"
    * "top customers by sales"
    * "highest sales by customer and product"
  you MUST:
    * pick an appropriate numeric metric column (e.g., VBRP.NETWR, BSAD.DMBTR, INVOICE_V2_BUSINESS_DATA.TOTAL_AMOUNT)
    * set "agg": "SUM" (or another relevant aggregate) on that metric column
    * give it a clear "description" like "total_sales" or "total_revenue"
    * add the dimension columns (industry, customer, product, country, etc.) to "group_by"
    * filter out NULL dimension values where it makes sense (e.g., industry IS NOT NULL)
    * order by the metric's description (e.g., "total_sales DESC") and use a small limit (e.g. 50 or 100).
- If the question is about "lowest", sort ASC instead of DESC.
- **CRITICAL for lowest/highest/top/bottom by dimension**: Exclude zero/empty aggregates.
  When grouping by customer, country, product, industry, etc. and showing SUM of sales/amounts,
  add "having": [{{ "lhs": "<metric_description>", "operator": ">", "rhs": "0" }}]
  so we only show entities that have actual activity. E.g. for "lowest sales by customer and country",
  add having on total_sales > 0 — otherwise we get customers with $0 (no sales), which is wrong.

- **MANDATORY: ALWAYS include currency** – whenever any monetary/amount column is selected
  (NETWR, WRBTR, DMBTR, HSL, WSL, KSL, KBETR, STPRS, WERTN, RMWWR, BRTWR, KBETR, etc.),
  you MUST also select the currency column. Rules by table (IMPORTANT – these are exact column names!):
  * VBRP or VBRK queries → add VBRK.WAERK (alias: "currency")   ← WAERK not WAERS for VBRK!
  * EKKO or EKPO queries → add EKKO.WAERS (alias: "currency")
  * RBKP or RSEG queries → add RBKP.WAERS (alias: "currency")
  * FAGLFLEXA queries → add FAGLFLEXA.RTCUR (alias: "currency") ← RTCUR not WAERS for FAGLFLEXA!
  * KEKO queries → add KEKO.HWAER (alias: "currency")            ← HWAER not WAERS for KEKO!
  * KONV queries → add KONV.WAERS (alias: "currency")
  * CKMLCR queries → add CKMLCR.WAERS (alias: "currency")
  Also add the currency column to group_by if group_by is non-empty.
  A number without a currency code is useless to the business user.

- **MANDATORY: ALWAYS include a date or period column** unless the question explicitly asks for
  a single grand-total number (e.g. "what is the total sales overall?"):
  * VBRK/VBRP queries → add VBRK.FKDAT (billing date, alias: "billing_date") to SELECT and group_by
  * RBKP queries → add RBKP.BUDAT (posting date, alias: "posting_date") to SELECT and group_by
  * EKKO queries → add EKKO.BEDAT (PO date, alias: "po_date") to SELECT and group_by
  * FAGLFLEXA queries → add FAGLFLEXA.POPER (posting period 01-12, alias: "period") AND
    FAGLFLEXA.RYEAR (fiscal year, alias: "fiscal_year") to SELECT and group_by
  This allows the user to see WHICH period the data belongs to.

- **MANDATORY: ALWAYS show MATERIAL NAME alongside material number** – raw codes are not useful:
  * Whenever MATNR appears in any table (VBRP, EKPO, KEKO, CKIS, VBAP, MARC, etc.),
    you MUST join MAKT and SELECT MAKT.MAKTX (description) with alias "material_name".
  * Add MAKT to tables list, join: <source_table>.MATNR = MAKT.MATNR
  * Add MAKT.MAKTX to group_by if group_by is non-empty.
  * If MAKT is already in the query, just make sure MAKTX is in the columns list.
  * Exception: if the question explicitly says "show material number" or "list MATNR codes".

- **TEXT SEARCH / FILTER by name or word**: When the question asks to filter by a word or name
  (e.g. "containing 'jacket'", "with word 'pump'", "products that include 'motor'",
  "customers named 'Smith'", "vendors containing 'GmbH'"), you MUST add an ILIKE filter:
  * For product/material names: filter on MAKT.MAKTX ILIKE '%<word>%'
    (always join MAKT if not already included)
  * For customer names: filter on KNA1.NAME1 ILIKE '%<word>%'
  * For vendor names: filter on LFA1.NAME1 ILIKE '%<word>%'
  * Do NOT skip this filter — returning all rows instead of the matching subset is wrong.
  * Example for "list all products containing 'jacket'":
    tables: [VBRP, VBRK, MAKT], join VBRP.MATNR = MAKT.MATNR,
    filter: MAKT.MAKTX ILIKE '%jacket%', SELECT MAKT.MAKTX, SUM(VBRP.FKIMG), SUM(VBRP.NETWR)

- **MANDATORY: NEVER add date column to GROUP BY on aggregated queries** (queries with SUM/AVG):
  For aggregated queries like "top N customers by total revenue", "sales by country", etc.,
  NEVER put FKDAT, BUDAT, or any date in group_by — that would fragment the total into
  per-day rows and give wrong results (e.g. revenue per customer per day instead of total).
  Instead, for aggregated queries, add date as MIN/MAX aggregates for time-range context:
  * {{"table": "VBRK", "name": "FKDAT", "description": "earliest_billing_date", "agg": "MIN"}}
  * {{"table": "VBRK", "name": "FKDAT", "description": "latest_billing_date", "agg": "MAX"}}
  Only add raw date to group_by for detail (non-aggregated) row-level queries.

- **CRITICAL – INDIVIDUAL INVOICE ROWS (DO NOT AGGREGATE)**: When the user asks to
  "show invoices", "list invoices", "display invoices with [customer/payer/industry]",
  or "show billing documents with [names/details]" — return INDIVIDUAL ROWS, NOT aggregated totals.
  * Do NOT use GROUP BY, SUM, or COUNT for invoice listing queries.
  * Select: VBRK.VBELN (invoice_number), VBRK.FKDAT (billing_date), VBRK.KUNAG (customer_number),
    KNA1.NAME1 (payer_name/customer_name), VBRK.NETWR (invoice_amount), VBRK.WAERK (currency)
  * If user asks "by industry": also join T016T and select COALESCE(T016T.brtxt, KNA1.brsch) as industry
  * Join: VBRK.KUNAG = KNA1.KUNNR (LEFT JOIN), KNA1.brsch = T016T.brsch (LEFT JOIN)
  * DO NOT add group_by, DO NOT add agg on netwr — just individual rows.
  * Limit: 200 rows. Order by billing_date DESC (or by industry ASC, billing_date DESC if asked by industry).
  * Example correct output for "show invoices with payer names by industry":
    tables: [VBRK, KNA1, T016T], no group_by, no agg,
    columns: VBRK.VBELN, VBRK.FKDAT, VBRK.KUNAG, KNA1.NAME1, T016T.brtxt, VBRK.NETWR, VBRK.WAERK
    order: industry ASC, billing_date DESC, limit: 200
"""
    messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
    spec: Dict[str, Any] = {}

    for _attempt in range(3):  # up to 3 attempts; extra rounds fix markdown-wrapped / truncated JSON
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,  # type: ignore[arg-type]
            temperature=0,
        )
        content = resp.choices[0].message.content or ""

        # ── Try direct parse ──────────────────────────────────────────────────
        parsed: Optional[Dict[str, Any]] = None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # LLM sometimes wraps in ```json ... ``` fences — strip and retry
            stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.DOTALL)
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                # Last-resort: grab the first {...} block
                m = re.search(r"\{.*\}", stripped, re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                    except json.JSONDecodeError:
                        pass

        if parsed:
            logger.info("_generate_sql_json: parsed spec on attempt %d", _attempt + 1)
            return parsed

        # ── Parse failed: log and ask the model to fix its output ────────────
        logger.warning(
            "_generate_sql_json: JSON parse failed (attempt %d/3). Response preview: %r",
            _attempt + 1,
            content[:400],
        )
        if _attempt < 2:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "Your previous response could not be parsed as JSON. "
                        "Return ONLY the raw JSON object — no markdown fences (no ```), "
                        "no explanatory text before or after, no comments. "
                        "Start your response with { and end with }."
                    ),
                },
            ]

    logger.error(
        "_generate_sql_json: all 3 attempts failed to produce valid JSON for question: %r",
        question,
    )
    return {}


def _ensure_having_for_aggregates(spec: Dict[str, Any], question: str) -> None:
    """
    When question asks for lowest/highest/top/bottom by dimension with aggregates,
    auto-inject HAVING metric > 0 so we exclude entities with $0 (wrong for rankings).
    Modifies spec in place.
    """
    q_lower = (question or "").lower()
    if not any(kw in q_lower for kw in ("lowest", "highest", "top", "bottom", "minimum", "maximum", "best", "worst")):
        return
    if spec.get("having"):
        return
    group_bys = spec.get("group_by") or []
    if not group_bys:
        return
    columns = spec.get("columns") or []
    sum_col = None
    for col in columns:
        if str(col.get("agg") or "").upper() in ("SUM", "AVG"):
            sum_col = col
            break
    if not sum_col:
        return
    human = sum_col.get("description") or f"{sum_col.get('table', '')}_{sum_col.get('name', '')}"
    human_safe = re.sub(r"[^\w]", "_", str(human))[:60] or "total"
    spec["having"] = [{"lhs": human_safe, "operator": ">", "rhs": "0"}]
    logger.info("Auto-injected HAVING %s > 0 for ranking query", human_safe)


def _auto_enrich_spec(spec: Dict[str, Any], question: str) -> None:
    """
    Post-process LLM-generated SQL spec to enforce three mandatory context columns:
      1. Currency (WAERS) whenever monetary amounts are present
      2. Date/period column so user knows WHEN the data is from
      3. MAKT.MAKTX alongside any MATNR column (human-readable material name)

    IMPORTANT date rule: date/period is added to SELECT for context, but NEVER to GROUP BY
    on aggregated queries (queries with SUM/AVG columns). Adding a date to GROUP BY on an
    aggregated query changes "top 20 customers by total revenue" into
    "revenue per customer per day" — completely wrong semantics.
    Instead, for aggregated queries we add MIN/MAX of the date so the user can see
    the time range covered without fragmenting the aggregation.

    Modifies spec in-place.  This is a safety net — the LLM prompt already asks for these,
    but we enforce them here in case the model skips one.
    """
    if not spec:
        return

    q_lower = (question or "").lower()

    # Resolve sets of tables and column names currently in the spec
    tables_in_spec = {(t.get("name") or "").upper() for t in spec.get("tables", [])}
    col_names_upper = {(c.get("name") or "").upper() for c in spec.get("columns", [])}

    # Build a map from UPPERCASE table name → actual name as used in spec (preserving LLM case)
    # This is used when adding enrichment columns so the table name in the added column matches
    # exactly what the LLM put in spec["tables"], avoiding validate_sql_spec case mismatches.
    _spec_table_actual_name: Dict[str, str] = {
        (t.get("name") or "").upper(): (t.get("name") or "")
        for t in spec.get("tables", []) if t.get("name")
    }
    def _spec_tbl(uppercase_key: str) -> str:
        """Return the table name as it appears in spec (preserving LLM case) or fallback to uppercase_key."""
        return _spec_table_actual_name.get(uppercase_key, uppercase_key)

    # Use schema_ai_config.json numeric_columns — no hardcoded list needed.
    # Adding a new numeric column to the config automatically flows through here.
    _cfg = _get_schema_config()
    AMOUNT_COLS = {c.upper() for c in (_cfg.get("numeric_columns") or [])}
    # Fallback in case config is empty (should not happen after our edits)
    if not AMOUNT_COLS:
        AMOUNT_COLS = {"NETWR", "WRBTR", "DMBTR", "HSL", "WSL", "KSL", "KBETR", "STPRS"}
    CURRENCY_COLS = {"WAERS", "WAERK", "RCUR", "RTCUR", "RWCUR", "HWAER", "FWAER_KPF"}
    DATE_PERIOD_COLS = {"FKDAT", "BUDAT", "BEDAT", "POPER", "GJAHR", "RYEAR", "BLDAT", "AUGDT"}

    has_amounts = bool(col_names_upper & AMOUNT_COLS)
    has_currency = bool(col_names_upper & CURRENCY_COLS)
    has_date_period = bool(col_names_upper & DATE_PERIOD_COLS)
    has_matnr = "MATNR" in col_names_upper
    has_maktx = "MAKTX" in col_names_upper
    has_group_by = bool(spec.get("group_by"))

    # Is there any aggregate column (SUM/AVG/COUNT/MIN/MAX)?  Critical for date rule.
    has_aggregate = any(
        str(c.get("agg") or "").upper() in {"SUM", "AVG", "COUNT", "MIN", "MAX"}
        for c in spec.get("columns", [])
    )

    # Is this a pure single-number grand-total query? Skip date enrichment for those.
    is_grand_total = any(
        phrase in q_lower for phrase in
        ("grand total", "overall total", "total overall", "all time total", "in total",
         "total amount", "how much total", "sum total")
    ) and not has_group_by

    # ─── 1. Currency enrichment ───────────────────────────────────────────────
    if has_amounts and not has_currency:
        CURRENCY_SOURCE = [
            # trigger_tbl, src_tbl, src_col (actual DB column name!), alias
            ("VBRK",       "VBRK",       "WAERK",  "currency"),   # VBRK uses WAERK not WAERS
            ("EKKO",       "EKKO",       "WAERS",  "currency"),
            ("RBKP",       "RBKP",       "WAERS",  "currency"),
            ("FAGLFLEXA",  "FAGLFLEXA",  "RTCUR",  "currency"),   # FAGLFLEXA uses RTCUR not WAERS
            ("KEKO",       "KEKO",       "HWAER",  "currency"),   # KEKO uses HWAER not WAERS
            ("VBRP",       "VBRK",       "WAERK",  "currency"),   # VBRP pulls currency from VBRK.WAERK
            ("RSEG",       "RBKP",       "WAERS",  "currency"),
            ("EKPO",       "EKKO",       "WAERS",  "currency"),
            ("KONV",       "KONV",       "WAERS",  "currency"),   # KONV has WAERS
            ("CKMLCR",     "CKMLCR",     "WAERS",  "currency"),   # CKMLCR has WAERS
        ]
        for trigger_tbl, src_tbl, src_col, alias in CURRENCY_SOURCE:
            if trigger_tbl in tables_in_spec and src_tbl in tables_in_spec:
                # Use the exact table name casing from spec (not hardcoded uppercase)
                # to avoid validate_sql_spec false-positive case-mismatch errors.
                actual_src_tbl = _spec_tbl(src_tbl)
                spec.setdefault("columns", []).append(
                    {"table": actual_src_tbl, "name": src_col, "description": alias, "agg": None}
                )
                # Currency is safe to GROUP BY — same value for all rows in a billing doc
                if has_group_by:
                    spec.setdefault("group_by", []).append({"table": actual_src_tbl, "column": src_col})
                logger.info("Auto-enriched spec: added %s.%s (currency)", actual_src_tbl, src_col)
                break

    # ─── 2. Date / period enrichment ─────────────────────────────────────────
    # RULE: For aggregated queries (has SUM/AVG), add MIN/MAX of the date as range markers
    # — never add a raw date to GROUP BY, which would fragment the aggregation.
    # For non-aggregated (detail) queries, add date normally (and to GROUP BY if needed).
    if not has_date_period and not is_grand_total:
        DATE_SOURCE = [
            ("VBRK",      "VBRK",      "FKDAT",  "billing_date"),
            ("RBKP",      "RBKP",      "BUDAT",  "posting_date"),
            ("EKKO",      "EKKO",      "BEDAT",  "po_date"),
            ("FAGLFLEXA", "FAGLFLEXA", "POPER",  "period"),
        ]
        for trigger_tbl, src_tbl, src_col, alias in DATE_SOURCE:
            if trigger_tbl in tables_in_spec and src_tbl in tables_in_spec:
                # Use the exact table name casing from spec
                actual_src_tbl = _spec_tbl(src_tbl)
                if has_aggregate:
                    # Aggregated query: add MIN/MAX as range markers (no GROUP BY change)
                    spec.setdefault("columns", []).append(
                        {"table": actual_src_tbl, "name": src_col,
                         "description": f"earliest_{alias}", "agg": "MIN"}
                    )
                    spec.setdefault("columns", []).append(
                        {"table": actual_src_tbl, "name": src_col,
                         "description": f"latest_{alias}", "agg": "MAX"}
                    )
                    logger.info(
                        "Auto-enriched spec: added MIN/MAX(%s.%s) date range for aggregated query",
                        actual_src_tbl, src_col
                    )
                else:
                    # Detail (non-aggregated) query: add date as plain column
                    spec.setdefault("columns", []).append(
                        {"table": actual_src_tbl, "name": src_col, "description": alias, "agg": None}
                    )
                    if has_group_by:
                        spec.setdefault("group_by", []).append({"table": actual_src_tbl, "column": src_col})
                    # FAGLFLEXA: also add RYEAR
                    if src_col == "POPER":
                        actual_faglflexa = _spec_tbl("FAGLFLEXA")
                        spec["columns"].append(
                            {"table": actual_faglflexa, "name": "RYEAR",
                             "description": "fiscal_year", "agg": None}
                        )
                        if has_group_by:
                            spec["group_by"].append({"table": actual_faglflexa, "column": "RYEAR"})
                    logger.info(
                        "Auto-enriched spec: added %s.%s (date/period, non-aggregated)",
                        actual_src_tbl, src_col
                    )
                break

    # ─── 3. Material name (MAKTX) enrichment ────────────────────────────────
    if has_matnr and not has_maktx:
        matnr_source_tables = [
            (c.get("table") or "").upper()
            for c in spec.get("columns", [])
            if (c.get("name") or "").upper() == "MATNR"
        ]
        source_tbl = matnr_source_tables[0] if matnr_source_tables else None

        if "MAKT" not in tables_in_spec and source_tbl:
            spec.setdefault("tables", []).append(
                {"name": "MAKT", "description": "Material descriptions (MAKTX)"}
            )
            spec.setdefault("joins", []).append({
                "left": source_tbl,
                "right": "MAKT",
                "on": f"{source_tbl}.MATNR = MAKT.MATNR"
            })

        spec.setdefault("columns", []).append(
            {"table": "MAKT", "name": "MAKTX", "description": "material_name", "agg": None}
        )
        if has_group_by:
            spec.setdefault("group_by", []).append({"table": "MAKT", "column": "MAKTX"})
        logger.info("Auto-enriched spec: added MAKT.MAKTX (material name) for table %s", source_tbl)


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
    # Build a map from logical name (any case) -> actual DB table name.
    #
    # IMPORTANT: We build from ALL tables in column_mappings (not just SAP_TABLE_DESCRIPTIONS)
    # so that any table in the DB (e.g. KONV, VBFA, AUFK, CKIS) is resolvable even if it
    # wasn't hardcoded in SAP_TABLE_DESCRIPTIONS.
    logical_to_actual: Dict[str, str] = {}

    # First pass: explicit SAP_TABLE_DESCRIPTIONS keys (backward compat, prefer these for aliases)
    for logical in SAP_TABLE_DESCRIPTIONS.keys():
        for actual in column_mappings.keys():
            if actual.lower() == logical.lower():
                logical_to_actual[logical] = actual
                logical_to_actual[logical.lower()] = actual
                logical_to_actual[logical.upper()] = actual

    # Second pass: every table that exists in column_mappings but wasn't mapped above.
    # This handles tables like KONV, VBFA, CEPC, COEP, CKIS etc. that are in the DB but
    # not in the hardcoded SAP_TABLE_DESCRIPTIONS dict.
    for actual in column_mappings.keys():
        if actual not in logical_to_actual:
            logical_to_actual[actual] = actual
        if actual.lower() not in logical_to_actual:
            logical_to_actual[actual.lower()] = actual
        if actual.upper() not in logical_to_actual:
            logical_to_actual[actual.upper()] = actual

    def _actual_table_name(logical: str) -> str:
        key = logical
        if key not in logical_to_actual:
            key = logical.lower()
        if key not in logical_to_actual:
            key = logical.upper()
        return logical_to_actual.get(key, logical)

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

    # SELECT (also build alias -> aggregate expression for HAVING; PostgreSQL does not allow SELECT aliases in HAVING)
    select_parts: List[str] = []
    used_col_aliases: set[str] = set()
    alias_to_agg_expr: Dict[str, str] = {}

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

        # Config-driven casting: schema_ai_config.json defines numeric_columns and trim_numeric_tables.
        # Add new tables/columns there when schema changes; no code changes needed.
        cfg = _get_schema_config()
        numeric_cols = {c.upper() for c in (cfg.get("numeric_columns") or [])}
        trim_tables = {t.upper() for t in (cfg.get("trim_numeric_tables") or [])}
        actual_upper = actual_tbl.upper()
        col_upper = str(col_name).upper()
        needs_numeric_cast = col_upper in numeric_cols
        use_trim_pattern = actual_upper in trim_tables and needs_numeric_cast

        if agg in {"SUM", "AVG", "COUNT", "MIN", "MAX"}:
            if agg == "COUNT":
                expr = f"{agg}({alias}.\"{col_name}\")"
            elif use_trim_pattern:
                # trim_numeric_tables: SAP tables where amounts are stored as VARCHAR/CHAR in PostgreSQL.
                # These need TRIM + NULLIF + ::numeric cast to handle empty strings.
                expr = f"{agg}(NULLIF(TRIM({alias}.\"{col_name}\"::text), '')::numeric)"
            else:
                # All other tables (EKPO, RBKP, RSEG, FAGLFLEXA, CKMLCR, COEP, BSAD, etc.)
                # have proper numeric/decimal column types in PostgreSQL.
                # Do NOT apply NULLIF(col, '') — that compares numeric to text and crashes with
                # "operator does not exist: numeric = text".
                # Just aggregate directly; PostgreSQL handles NULLs automatically.
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
        if agg in {"SUM", "AVG", "COUNT", "MIN", "MAX"}:
            alias_to_agg_expr[human_safe] = expr
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

    # Helper to rewrite "VBRP.VBELN" → "v.\"vbeln\"" using aliases and actual DB column names.
    # Postgres identifiers are case-sensitive when quoted; DB usually has lowercase columns.
    def _rewrite_expr(expr: str) -> str:
        out = expr
        # Replace Table.Column with alias."actual_column" (uses real DB column casing)
        def _repl(m: re.Match) -> str:
            tbl_part = m.group(1)
            col_part = m.group(2)
            actual_tbl = _actual_table_name(tbl_part)
            alias = table_aliases.get(actual_tbl)
            if not alias:
                return m.group(0)
            actual_col = _actual_column_name(actual_tbl, col_part)
            return f'{alias}."{actual_col}"'
        out = re.sub(r'\b(\w+)\.(\w+)\b', _repl, out)
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

    # Add any missing tables with heuristic joins on common keys (skip if no common key; else invalid SQL)
    COMMON_KEYS = ["VBELN", "KUNNR", "KUNAG", "MATNR", "EBELN", "LIFNR", "BELNR", "BRSCH"]
    for logical_tbl in all_tables:
        actual_tbl = _actual_table_name(logical_tbl)
        if actual_tbl in added_actuals:
            continue
        alias = table_aliases[actual_tbl]
        # Try to join on a shared key with an already-added table
        join_cond = None
        for key in COMMON_KEYS:
            other_col = _actual_column_name(actual_tbl, key)
            if not other_col:
                continue
            for added in added_actuals:
                base_col = _actual_column_name(added, key)
                if base_col:
                    add_alias = table_aliases.get(added)
                    if add_alias:
                        join_cond = f'{add_alias}."{base_col}" = {alias}."{other_col}"'
                        break
            if join_cond:
                break
        if join_cond:
            sql_lines.append(f'\nLEFT JOIN "{actual_tbl}" AS {alias}')
            sql_lines.append(f"    ON {join_cond}")
            added_actuals.add(actual_tbl)

    # WHERE (must come BEFORE GROUP BY)
    conds: List[str] = []
    for f in json_spec.get("filters", []) or []:
        lhs = _rewrite_expr(str(f.get("lhs", "")))
        op = str(f.get("operator", "")).strip().upper()
        rhs_raw = f.get("rhs")
        
        if not lhs or not op:
            continue
        
        # Handle NULL operators (don't need RHS)
        if op in {"IS NULL", "IS NOT NULL"}:
            conds.append(f"{lhs} {op}")
        else:
            # Need RHS for all other operators
            rhs = str(rhs_raw).strip() if rhs_raw is not None else ""
            if rhs and rhs.lower() != "none":
                conds.append(f"{lhs} {op} {rhs}")

    # NOTE: Do NOT auto-add IS NOT NULL conditions here.
    # The auto-IS-NOT-NULL for KNA1 was converting LEFT JOINs to INNER JOINs,
    # causing 0 rows whenever KNA1 data is incomplete (e.g. partial test datasets).
    # The LLM prompt instructs explicit NULL filtering when needed for a specific query.

    if conds:
        sql_lines.append("\nWHERE " + " AND ".join(conds))

    # GROUP BY (must come AFTER WHERE)
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

    # HAVING (after GROUP BY, before ORDER BY)
    # PostgreSQL does NOT allow SELECT aliases in HAVING; use the full aggregate expression instead.
    having_parts: List[str] = []
    for h in json_spec.get("having", []) or []:
        if not isinstance(h, dict):
            continue
        lhs = str(h.get("lhs", "")).strip()
        op = str(h.get("operator", "")).strip().upper()
        rhs_raw = h.get("rhs")
        if not lhs or not op:
            continue
        # Match alias by name (case-insensitive)
        lhs_lower = lhs.lower()
        matched = lhs if lhs in used_col_aliases else None
        if not matched:
            for a in used_col_aliases:
                if a.lower() == lhs_lower or lhs_lower in a.lower():
                    matched = a
                    break
        # Use aggregate expression if available; otherwise fall back to alias (may fail in PG)
        lhs_expr = alias_to_agg_expr.get(matched) if matched else None
        if not lhs_expr and matched:
            lhs_expr = f'"{matched}"'
        elif not lhs_expr:
            lhs_expr = lhs
        # Ensure any raw table.column inside HAVING is rewritten to the correct alias/column
        lhs_expr = _rewrite_expr(lhs_expr)
        if op in {"IS NULL", "IS NOT NULL"}:
            having_parts.append(f"{lhs_expr} {op}")
        else:
            rhs = str(rhs_raw).strip() if rhs_raw is not None else ""
            if rhs and rhs.lower() != "none":
                having_parts.append(f"{lhs_expr} {op} {rhs}")
    if having_parts:
        sql_lines.append("\nHAVING " + " AND ".join(having_parts))

    # ORDER BY
    order_by_parts: List[str] = []
    for ob in json_spec.get("order_by", []) or []:
        if isinstance(ob, str):
            # Check if it's a SELECT alias first
            ob_clean = ob.strip().split()[0]  # Remove DESC/ASC
            if ob_clean in used_col_aliases:
                order_by_parts.append(ob)
            else:
                order_by_parts.append(_rewrite_expr(ob))
        else:
            t_logical = ob.get("table")
            col_raw = ob.get("column")
            direction = ob.get("direction", "DESC").upper()
            
            # Try to match against SELECT aliases first (case-insensitive)
            col_lower = str(col_raw).lower().strip() if col_raw else ""
            matched_alias = None
            for used_alias in used_col_aliases:
                if col_lower in used_alias.lower() or used_alias.lower() in col_lower:
                    matched_alias = used_alias
                    break
            
            if matched_alias:
                # Use the SELECT alias directly
                order_by_parts.append(f'"{matched_alias}" {direction}')
            elif t_logical and col_raw:
                # Fall back to table.column format
                t_actual = _actual_table_name(t_logical)
                alias = table_aliases.get(t_actual)
                if alias and t_actual:
                    col = _actual_column_name(t_actual, col_raw)
                    order_by_parts.append(f'{alias}."{col}" {direction}')
    
    if order_by_parts:
        sql_lines.append("\nORDER BY " + ", ".join(order_by_parts))

    sql_lines.append(f"\nLIMIT {limit}")
    return "\n".join(sql_lines) + ";"


def _run_sql(db: Session, sql: str) -> List[Dict[str, Any]]:
    """Execute SQL and return rows. Raises exception on failure so the retry loop
    receives the REAL Postgres error (e.g. 'column X does not exist') instead of
    a misleading 'no rows' message that causes the LLM to generate a wrong refinement."""
    if not sql or not sql.strip():
        return []
    # Intentionally NOT catching exceptions here.  Callers (run_sap_sql_agent) have a
    # try/except that captures the real error message and passes it to refine_query_on_error.
    result = db.execute(text(sql))
    rows = result.fetchall()
    keys = result.keys()
    out: List[Dict[str, Any]] = []
    for row in rows:
        row_dict = {k: _serialize_value(v) for k, v in zip(keys, row)}
        out.append(row_dict)
    return out


def run_purchase_order_fallback(db: Session, question: str) -> Optional[SqlAgentResult]:
    """
    When the main agent fails to generate SQL for a purchase-order question, run a direct
    EKPO aggregate: total quantity and total cost by material. Uses introspected table/column names.
    """
    if not _is_purchase_order_question(question):
        return None
    try:
        insp = inspect(db.bind)
        all_tables = insp.get_table_names()
        ekpo_table = None
        for t in all_tables:
            if (t or "").upper() == "EKPO":
                ekpo_table = t
                break
        if not ekpo_table:
            logger.warning("run_purchase_order_fallback: EKPO table not found")
            return None
        cols = [c["name"] for c in insp.get_columns(ekpo_table)]
        cols_lower = {c.lower(): c for c in cols}
        matnr_col = cols_lower.get("matnr")
        menge_col = cols_lower.get("menge")
        netpr_col = cols_lower.get("netpr")
        if not all((matnr_col, menge_col, netpr_col)):
            logger.warning("run_purchase_order_fallback: EKPO missing matnr/menge/netpr columns: %s", cols)
            return None
        # Quote identifiers for Postgres (case-sensitive)
        def q(s: str) -> str:
            return f'"{s}"' if s and s != s.lower() else (s or "")
        sql = (
            f'SELECT {q(matnr_col)} AS material, '
            f'SUM({q(menge_col)}) AS total_quantity, '
            f'SUM({q(menge_col)} * {q(netpr_col)}) AS total_cost '
            f'FROM {q(ekpo_table)} '
            f'GROUP BY {q(matnr_col)} '
            f'ORDER BY total_cost DESC NULLS LAST LIMIT 100'
        )
        rows = _run_sql(db, sql)
        if not rows:
            return None
        logger.info("run_purchase_order_fallback: executed EKPO aggregate, %d rows", len(rows))
        return SqlAgentResult(sql=sql + ";", rows=rows)
    except Exception as e:
        logger.warning("run_purchase_order_fallback failed: %s", e)
        return None


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
- Explain the answer using ONLY the exact numbers and values from the JSON above.
- Do NOT invent, approximate, or reuse numbers from memory or prior context.
- Include specific numbers (totals, top items, customers, countries, industries) from the data.
- **Always state the currency** of any monetary amounts (e.g. "USD", "EUR"). If a "currency",
  "WAERS", or "WAERK" column is present in the data, use it. If not, note the currency is unknown.
- **Always state the time period** the data covers. If "billing_date", "FKDAT", "posting_date",
  "BUDAT", "period", "POPER", "fiscal_year", or "RYEAR" columns are present, mention the date range
  or period. If no date column is present, note the time scope (e.g. "all available periods").
- For material numbers (MATNR), always use the material name (MAKTX) instead of the raw code
  if a "material_name" or "MAKTX" column is present in the data.
- Be concise (3–8 sentences).
- If the JSON is empty, say clearly that no data was returned for this query.
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
    
    # Validate columns reference existing tables (case-insensitive: LLM may mix VBRK/vbrk)
    table_names = {t.get("name") for t in tables if isinstance(t, dict) and t.get("name")}
    table_names_lower = {(n or "").lower() for n in table_names}
    for col in columns:
        if isinstance(col, dict):
            col_table = col.get("table")
            if col_table and (col_table not in table_names) and (col_table.lower() not in table_names_lower):
                errors.append(f"Column references unknown table: {col_table}")

    # Validate joins reference existing tables (case-insensitive)
    joins = spec.get("joins", [])
    for j in joins:
        if isinstance(j, dict):
            left = j.get("left")
            right = j.get("right")
            if left and (left not in table_names) and (left.lower() not in table_names_lower):
                errors.append(f"Join references unknown left table: {left}")
            if right and (right not in table_names) and (right.lower() not in table_names_lower):
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
- Invalid column names (use exact DB column names; Postgres columns are usually lowercase)
- Incorrect table references
- Missing GROUP BY for aggregated columns
- T016T has only brsch/brtxt columns — NEVER join T016T on VBELN; only join KNA1.brsch = T016T.brsch when question asks about industry
- Do NOT include T016T unless the question asks about industry
- Query returned no rows: remove date filters, use all periods, simplify to fewer joins
- Results include $0 / zero aggregates: add "having": [{{ "lhs": "<metric_alias>", "operator": ">", "rhs": "0" }}]

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


def run_schema_driven_sql_agent(
    question: str,
    db: Session,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
) -> SqlAgentResult | None:
    """
    Schema-driven SQL agent: no keyword rules. Flow is:
    1) Load schema from DB
    2) LLM selects relevant tables from schema
    3) LLM generates PostgreSQL SQL from question + tables + schema
    4) Validate SQL against schema, execute, return rows.

    Use this first; fall back to run_adaptive_sap_sql_agent / run_sap_sql_agent on failure.
    """
    if not _SCHEMA_DRIVEN_AVAILABLE:
        return None
    client = _get_openai_client()
    if not client:
        return None
    try:
        schema = get_schema_dict(db)
        if not schema:
            logger.warning("schema_driven_agent: no schema loaded")
            return None
        available_tables = list(schema.keys())
        schema_table_case = {t.upper(): t for t in available_tables}

        # Semantic dictionary fast path: resolve metric+dimension and build template SQL
        template_sql = None
        if try_resolve_and_build_sql:
            template_sql = try_resolve_and_build_sql(
                question,
                available_tables=available_tables,
                schema_table_case=schema_table_case,
            )
        if template_sql:
            is_valid, err = schema_validate_sql(template_sql, schema)
            if is_valid:
                rows = _run_sql(db, template_sql)
                return SqlAgentResult(sql=template_sql, rows=rows)
            logger.debug("schema_driven_agent: template SQL invalid (%s), falling back to LLM", err)

        # Use get_schema_text so table selector sees semantic map (vendor→LFA1, delivery→LIKP/LIPS, etc.)
        schema_text = get_schema_text(db, include_semantic_map=True)
        tables = schema_select_tables(question, schema_text, client, available_tables)
        # Intent-based table fallback when LLM returns no tables
        if not tables:
            available_upper = {t.upper(): t for t in available_tables}
            if _is_purchase_order_question(question):
                tables = [available_upper[t] for t in ("EKPO", "MAKT") if t in available_upper]
                if tables:
                    logger.info("schema_driven_agent: purchase-order intent fallback tables: %s", tables)
            elif _is_internal_order_question(question):
                # AUFK for order list; COEP for cost by order (join on objnr)
                tables = [available_upper[t] for t in ("AUFK", "COEP") if t in available_upper]
                if not tables:
                    tables = [available_upper["AUFK"]] if "AUFK" in available_upper else []
                if tables:
                    logger.info("schema_driven_agent: internal-order intent fallback tables: %s", tables)
        if not tables:
            logger.warning("schema_driven_agent: no tables selected for question: %s", (question or "")[:80])
            return None
        logger.info("schema_driven_agent: selected_tables=%s", tables)
        schema_subset = schema_to_text(schema, table_subset=tables)
        # Optional: similar past queries as few-shot examples
        similar: Optional[List[tuple]] = None
        if few_shot_examples:
            similar = [(ex.get("user_query") or "", ex.get("sql_query") or "") for ex in few_shot_examples if ex.get("sql_query")]
        sql = schema_generate_sql(question, tables, schema_subset, client, similar_examples=similar)
        if not sql:
            logger.warning("schema_driven_agent: no SQL generated for question: %s", (question or "")[:80])
            return None
        logger.info("schema_driven_agent: generated_sql (first 300 chars): %s", (sql or "")[:300])
        is_valid, err = schema_validate_sql(sql, schema)
        if not is_valid:
            logger.warning("schema_driven_agent: SQL validation failed: %s", err)
            return None
        rows = _run_sql(db, sql)
        return SqlAgentResult(sql=sql, rows=rows)
    except Exception as e:
        logger.warning("schema_driven_agent failed: %s", e)
        return None


def run_sap_sql_agent(
    question: str,
    db: Session,
    knowledge_context: Optional[str] = None,
    max_retries: int = 2,
    time_scope: str = "current",
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
) -> SqlAgentResult | None:
    """
    Main entry point used by the dashboard AI endpoint.

    It:
    - Uses LLM to pick tables (dynamically from DB + mapping; supports new tables)
    - Introspects columns from Postgres
    - Uses LLM to build a JSON SQL spec
    - Translates to Postgres SQL and executes
    - Caches SQL per question

    Args:
        knowledge_context: Optional user preferences (e.g. "For cost queries use EKPO, RBKP, RSEG").
        time_scope: 'current' (recent data), 'historical' (1994-2010), or 'both' (all periods)
    """
    client = _get_openai_client()
    if not client:
        return None

    # ── CATALOG FAST-PATH ────────────────────────────────────────────────────
    # Try the pre-built SQL catalog first — no LLM calls, instant, reliable.
    # Falls through to the LLM path if no confident match or SQL returns 0 rows.
    try:
        catalog_sql = _lookup_sql_catalog(question)
        if catalog_sql:
            # Fix table name quoting for PostgreSQL (catalog uses bare names like VBRK;
            # PostgreSQL requires "VBRK" for uppercase-quoted tables)
            catalog_sql_pg = _quote_catalog_sql_tables(catalog_sql)
            logger.debug("sql_catalog: executing SQL (after quoting):\n%s", catalog_sql_pg[:500])
            catalog_rows = _run_sql(db, catalog_sql_pg)
            if catalog_rows:
                logger.info(
                    "sql_catalog: FAST-PATH hit — %d rows returned for: %r",
                    len(catalog_rows), question[:80],
                )
                return SqlAgentResult(sql=catalog_sql_pg, rows=catalog_rows)
            else:
                logger.info(
                    "sql_catalog: matched but returned 0 rows — falling through to LLM path for: %r",
                    question[:80],
                )
    except Exception as _cat_err:
        logger.warning("sql_catalog: fast-path error (%s) — falling through to LLM path", _cat_err)
        try:
            db.rollback()
        except Exception:
            pass
    # ── END CATALOG FAST-PATH ────────────────────────────────────────────────

    # ── SEMANTIC DICTIONARY FAST-PATH ────────────────────────────────────────
    # Use entities + metrics + join graph to resolve question → SQL without LLM.
    if _SCHEMA_DRIVEN_AVAILABLE:
        try:
            schema = get_schema_dict(db)
            if schema:
                available = list(schema.keys())
                sem_sql = semantic_resolve_to_sql(question, available_tables=available)
                if not sem_sql:
                    sem_sql = semantic_resolve_count_by(question, available_tables=available)
                if sem_sql:
                    sem_sql_pg = _quote_catalog_sql_tables(sem_sql)
                    sem_rows = _run_sql(db, sem_sql_pg)
                    if sem_rows:
                        logger.info(
                            "semantic_sql_resolver: FAST-PATH hit — %d rows for: %r",
                            len(sem_rows), question[:80],
                        )
                        return SqlAgentResult(sql=sem_sql_pg, rows=sem_rows)
        except Exception as _sem_err:
            logger.debug("semantic_sql_resolver: %s — falling through to LLM path", _sem_err)
            try:
                db.rollback()
            except Exception:
                pass
    # ── END SEMANTIC DICTIONARY FAST-PATH ────────────────────────────────────

    attempt = 0
    last_error = None
    spec = None

    try:
        selected_tables = _pick_tables(question, client, db, knowledge_context)
        logger.info("sap_sql_agent: question='%s' | selected_tables=%s", question, selected_tables)

        column_mappings = _introspect_columns(db, selected_tables)
        if not column_mappings:
            logger.warning("sap_sql_agent: no column mappings found for selected tables %s", selected_tables)
            return None

        logger.info(
            "sap_sql_agent: usable_tables_after_introspection=%s",
            list(column_mappings.keys()),
        )

        table_descriptions = _get_table_descriptions(db)
        spec = _generate_sql_json(
            question,
            selected_tables,
            column_mappings,
            client,
            time_scope=time_scope,
            few_shot_examples=few_shot_examples,
            table_descriptions=table_descriptions,
        )
        if not spec:
            logger.error(
                "sap_sql_agent: LLM returned empty/invalid JSON spec after 3 attempts "
                "for question '%s' — cannot generate SQL. "
                "Check _generate_sql_json logs above for the raw LLM output.",
                question,
            )
            return None

        _ensure_having_for_aggregates(spec, question)
        _auto_enrich_spec(spec, question)  # enforce: currency, date/period, material name

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
        rows: List[Dict[str, Any]] = []
        sql = ""
        while attempt <= max_retries:
            try:
                sql = _json_to_sql_postgres(spec, column_mappings)
                logger.info(f"📝 Generated SQL:\n{sql}")
                rows = _run_sql(db, sql)

                if rows:
                    logger.info(f"✅ SQL returned {len(rows)} rows")
                    return SqlAgentResult(sql=sql, rows=rows)

                # Query returned no rows — retry with simpler query
                logger.warning(f"⚠️ SQL returned no rows for question: {question}")
                logger.warning(f"📊 SQL query:\n{sql}")
                if attempt < max_retries:
                    try:
                        db.rollback()
                    except Exception:
                        pass
                    logger.warning("Query returned 0 rows — refining SQL...")
                    spec = refine_query_on_error(
                        client,
                        question,
                        "Query returned no rows. The database may have historical data (1994-2010) but little recent data. "
                        "Remove date filters, use ALL periods, simplify joins to only essential tables.",
                        spec,
                    )
                    attempt += 1
                    continue
                break

            except Exception as sql_err:
                last_error = str(sql_err)
                logger.warning(f"SQL execution failed (attempt {attempt + 1}/{max_retries + 1}): {last_error}")
                
                if attempt < max_retries:
                    try:
                        db.rollback()
                    except Exception:
                        pass
                    logger.info("Refining query specification...")
                    spec = refine_query_on_error(client, question, last_error, spec)
                    attempt += 1
                else:
                    # Max retries reached
                    logger.error(f"Max retries reached for question: {question}")
                    return None

        # ADAPTIVE: If 0 rows and we used "current" (recent) scope, retry once with ALL periods.
        # Works for any question — sales, costs, compare — no hardcoding.
        if not rows and time_scope == "current":
            try:
                db.rollback()
            except Exception:
                pass
            logger.info("Retrying with time_scope='both' (all periods) — adaptive fallback")
            retry_result = run_sap_sql_agent(
                question,
                db,
                knowledge_context=knowledge_context,
                max_retries=1,
                time_scope="both",
                few_shot_examples=few_shot_examples,
            )
            if retry_result and retry_result.rows:
                return retry_result

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
