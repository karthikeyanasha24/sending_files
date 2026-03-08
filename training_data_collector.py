"""
Training data collection service for fine-tuning AI models.

Logs successful queries, SQL, results, and user feedback to build a training dataset.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def ensure_training_data_table(db: Session) -> None:
    """
    Create the training data table if it doesn't exist.
    Schema: user_id, user_query, sql_query, result_summary, feedback_score, metadata, created_at
    """
    try:
        db.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS ai_training_data (
                  id BIGSERIAL PRIMARY KEY,
                  user_id BIGINT NOT NULL,
                  user_query TEXT NOT NULL,
                  sql_query TEXT NULL,
                  result_summary TEXT NULL,
                  action_type VARCHAR(50) NULL,
                  feedback_score INTEGER NULL CHECK (feedback_score >= 1 AND feedback_score <= 5),
                  feedback_comment TEXT NULL,
                  metadata JSONB NULL,
                  created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                  feedback_at TIMESTAMPTZ NULL
                )
                """
            )
        )
        db.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_ai_training_data_user_id 
                ON ai_training_data(user_id)
                """
            )
        )
        db.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_ai_training_data_feedback_score 
                ON ai_training_data(feedback_score) 
                WHERE feedback_score IS NOT NULL
                """
            )
        )
        db.commit()
        logger.info("Training data table ensured")
    except Exception as e:
        logger.error(f"Failed to ensure training data table: {e}")
        db.rollback()


def log_query_execution(
    db: Session,
    user_id: int,
    user_query: str,
    sql_query: Optional[str],
    result_summary: str,
    action_type: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """
    Log a query execution to the training data table.
    
    Args:
        db: Database session
        user_id: User ID
        user_query: Natural language query from user
        sql_query: Generated SQL query (if any)
        result_summary: AI-generated summary of results
        action_type: Action taken (new, reuse, compare, follow-up, knowledge)
        metadata: Additional metadata (rows count, execution time, etc.)
    
    Returns:
        ID of the inserted record, or None on failure
    """
    try:
        ensure_training_data_table(db)
        
        result = db.execute(
            text(
                """
                INSERT INTO ai_training_data 
                (user_id, user_query, sql_query, result_summary, action_type, metadata, created_at)
                VALUES (:user_id, :user_query, :sql_query, :result_summary, :action_type, :metadata, :created_at)
                RETURNING id
                """
            ),
            {
                "user_id": user_id,
                "user_query": user_query[:5000],  # Limit query length
                "sql_query": sql_query[:10000] if sql_query else None,  # Limit SQL length
                "result_summary": result_summary[:10000],  # Limit summary length
                "action_type": action_type,
                "metadata": json.dumps(metadata or {}, default=str),
                "created_at": _now_utc(),
            },
        )
        db.commit()
        
        record_id = result.scalar()
        logger.info(f"Logged training data: record_id={record_id}, user_id={user_id}")
        return record_id
    
    except Exception as e:
        logger.error(f"Failed to log training data: {e}")
        db.rollback()
        return None


def submit_feedback(
    db: Session,
    record_id: int,
    feedback_score: int,
    feedback_comment: Optional[str] = None,
) -> bool:
    """
    Submit user feedback for a training data record.
    
    Args:
        db: Database session
        record_id: ID of the training data record
        feedback_score: Rating from 1-5 (1=poor, 5=excellent)
        feedback_comment: Optional text feedback
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if feedback_score < 1 or feedback_score > 5:
            logger.warning(f"Invalid feedback score: {feedback_score}")
            return False
        
        db.execute(
            text(
                """
                UPDATE ai_training_data
                SET feedback_score = :feedback_score,
                    feedback_comment = :feedback_comment,
                    feedback_at = :feedback_at
                WHERE id = :record_id
                """
            ),
            {
                "record_id": record_id,
                "feedback_score": feedback_score,
                "feedback_comment": feedback_comment[:2000] if feedback_comment else None,
                "feedback_at": _now_utc(),
            },
        )
        db.commit()
        logger.info(f"Submitted feedback for record_id={record_id}, score={feedback_score}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        db.rollback()
        return False


def export_training_dataset(
    db: Session,
    min_feedback_score: int = 4,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Export training data in OpenAI fine-tuning format (JSONL).
    
    Format:
    {
      "messages": [
        {"role": "system", "content": "You are an SAP data analyst..."},
        {"role": "user", "content": "Show top customers..."},
        {"role": "assistant", "content": "SQL: SELECT ... Summary: ..."}
      ]
    }
    
    Args:
        db: Database session
        min_feedback_score: Minimum feedback score to include (default: 4)
        limit: Maximum number of records to export
    
    Returns:
        List of training examples in OpenAI format
    """
    try:
        ensure_training_data_table(db)
        
        rows = db.execute(
            text(
                """
                SELECT user_query, sql_query, result_summary, action_type, feedback_score
                FROM ai_training_data
                WHERE feedback_score >= :min_score
                  AND sql_query IS NOT NULL
                  AND result_summary IS NOT NULL
                ORDER BY feedback_score DESC, created_at DESC
                LIMIT :limit
                """
            ),
            {"min_score": min_feedback_score, "limit": limit},
        ).fetchall()
        
        training_examples = []
        system_message = """You are an expert SAP data analyst assistant. You help users analyze sales, finance, and logistics data by:
1. Understanding their natural language questions
2. Generating accurate SQL queries for SAP-style tables
3. Providing clear, business-focused summaries of results

When given a question, generate SQL and explain the results concisely."""
        
        for row in rows:
            user_query = row[0]
            sql_query = row[1]
            result_summary = row[2]
            
            # Format assistant response: SQL + Summary
            assistant_content = f"```sql\n{sql_query}\n```\n\n{result_summary}"
            
            training_examples.append({
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": assistant_content},
                ]
            })
        
        logger.info(f"Exported {len(training_examples)} training examples")
        return training_examples
    
    except Exception as e:
        logger.error(f"Failed to export training dataset: {e}")
        return []


def get_few_shot_examples(
    db: Session,
    limit: int = 3,
    user_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch recent successful query→SQL examples for few-shot prompting.
    Prefers high feedback_score, then recent. Used to train the model by example.

    Returns:
        List of {"user_query": str, "sql_query": str} for injection into prompts.
    """
    try:
        ensure_training_data_table(db)
        where = "WHERE sql_query IS NOT NULL AND LENGTH(sql_query) > 20"
        params: Dict[str, Any] = {"limit": limit}
        if user_id is not None:
            where += " AND user_id = :user_id"
            params["user_id"] = user_id

        rows = db.execute(
            text(
                f"""
                SELECT user_query, sql_query
                FROM ai_training_data
                {where}
                ORDER BY COALESCE(feedback_score, 0) DESC, created_at DESC
                LIMIT :limit
                """
            ),
            params,
        ).fetchall()

        return [
            {"user_query": row[0] or "", "sql_query": row[1] or ""}
            for row in rows
            if row[0] and row[1]
        ]
    except Exception as e:
        logger.warning(f"Failed to fetch few-shot examples: {e}")
        return []


def get_training_stats(db: Session, user_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Get statistics about training data collection.
    
    Args:
        db: Database session
        user_id: Optional user ID to filter stats
    
    Returns:
        Dictionary with training data statistics
    """
    try:
        ensure_training_data_table(db)
        
        where_clause = "WHERE user_id = :user_id" if user_id else ""
        params = {"user_id": user_id} if user_id else {}
        
        result = db.execute(
            text(
                f"""
                SELECT 
                    COUNT(*) as total_queries,
                    COUNT(feedback_score) as rated_queries,
                    AVG(feedback_score) as avg_rating,
                    COUNT(CASE WHEN feedback_score >= 4 THEN 1 END) as high_quality_queries,
                    COUNT(CASE WHEN sql_query IS NOT NULL THEN 1 END) as queries_with_sql
                FROM ai_training_data
                {where_clause}
                """
            ),
            params,
        ).fetchone()
        
        return {
            "total_queries": int(result[0] or 0),
            "rated_queries": int(result[1] or 0),
            "avg_rating": float(result[2] or 0),
            "high_quality_queries": int(result[3] or 0),
            "queries_with_sql": int(result[4] or 0),
        }
    
    except Exception as e:
        logger.error(f"Failed to get training stats: {e}")
        return {
            "total_queries": 0,
            "rated_queries": 0,
            "avg_rating": 0.0,
            "high_quality_queries": 0,
            "queries_with_sql": 0,
        }
