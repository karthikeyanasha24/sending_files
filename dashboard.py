"""
Dashboard API endpoints for statistics, analytics, and AI insights
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, File, UploadFile, Body
from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Date, Numeric, inspect, text
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import json
import os

from ..database import get_db
from ..models.user import ZodiacUser
from ..models.invoice import ZodiacInvoiceSuccessEdi as SuccessModel, ZodiacInvoiceFailedEdi as FailedModel
from ..models.correction_cache import CorrectionCache
from ..models.invoice_business_data import InvoiceBusinessData
from ..models.invoice_v2_business_data import InvoiceV2BusinessData
from ..models.invoice_v2_validated import InvoiceV2Validated
from ..models.invoice_v2_document import InvoiceV2Document
from ..models.sat_simple_merged import SATSimpleMerged
from ..models.sat_document import SATDocument
from ..models.supplier_token import SupplierToken
from ..models.converted_invoice import ConvertedInvoice
from ..api.auth import get_current_user
from ..config.config import OPENAI_API_KEY, USE_SAP_DB_FOR_AI
from ..database import get_sap_session
from ..services.database import extract_supplier_info_from_string
from ..services.file_service import read_file_from_storage
from ..services.invoice_v2_business_intelligence import InvoiceV2BusinessIntelligence
from ..services.sap_sql_agent import answer_with_sap_sql_agent
from collections import defaultdict
from decimal import Decimal

# OpenAI client (optional)
try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False

router = APIRouter(prefix="/dashboard", tags=["dashboard"])
logger = logging.getLogger("zodiac-api.dashboard")


@router.get("/statistics")
async def get_dashboard_statistics(
    current_user: ZodiacUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    days: int = 30  # Default to last 30 days
):
    """
    Get comprehensive dashboard statistics including:
    - Overview (total, successful, failed, success rate)
    - Timeline data (invoices per day)
    - Format distribution
    - Customer distribution
    - Request type distribution (Web vs API)
    - Recent activity
    """
    try:
        logger.info(f"📊 Fetching dashboard statistics for user {current_user.id} (last {days} days)")
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # ============================================================
        # 1. OVERVIEW STATISTICS (Outbound Invoices)
        # ============================================================
        successful_count = db.query(SuccessModel).filter(
            SuccessModel.user_id == current_user.id,
            SuccessModel.deleted_at.is_(None)
        ).count()
        
        failed_count = db.query(FailedModel).filter(
            FailedModel.user_id == current_user.id,
            FailedModel.deleted_at.is_(None)
        ).count()
        
        total_count = successful_count + failed_count
        success_rate = (successful_count / total_count * 100) if total_count > 0 else 0
        
        # ============================================================
        # 1b. SAT DOCUMENT STATISTICS (Inbound)
        # ============================================================
        try:
            # Use SATSimpleMerged which tracks sent_to_sap status
            # Total SAT merged documents (each represents a batch sent to SAP)
            sat_total = db.query(func.count(SATSimpleMerged.id)).filter(
                SATSimpleMerged.user_id == current_user.id
            ).scalar() or 0
            
            # SAT documents successfully sent to SAP
            sat_sent_count = db.query(func.count(SATSimpleMerged.id)).filter(
                SATSimpleMerged.user_id == current_user.id,
                SATSimpleMerged.sent_to_sap == True
            ).scalar() or 0
            
            # SAT documents pending (not yet sent)
            sat_pending_count = db.query(func.count(SATSimpleMerged.id)).filter(
                SATSimpleMerged.user_id == current_user.id,
                SATSimpleMerged.sent_to_sap == False
            ).scalar() or 0
            
            logger.info(f"📄 SAT Stats: {sat_total} total, {sat_sent_count} sent to SAP, {sat_pending_count} pending")
        except Exception as sat_err:
            logger.warning(f"⚠️ Could not fetch SAT statistics: {sat_err}")
            sat_total = 0
            sat_sent_count = 0
            sat_pending_count = 0
        
        # ============================================================
        # 2. TIMELINE DATA (Invoices per day)
        # ============================================================
        # Successful invoices timeline
        success_timeline = db.query(
            cast(SuccessModel.uploaded_at, Date).label('date'),
            func.count(SuccessModel.id).label('count')
        ).filter(
            SuccessModel.user_id == current_user.id,
            SuccessModel.deleted_at.is_(None),
            SuccessModel.uploaded_at >= start_date
        ).group_by(cast(SuccessModel.uploaded_at, Date)).all()
        
        # Failed invoices timeline
        failed_timeline = db.query(
            cast(FailedModel.uploaded_at, Date).label('date'),
            func.count(FailedModel.id).label('count')
        ).filter(
            FailedModel.user_id == current_user.id,
            FailedModel.deleted_at.is_(None),
            FailedModel.uploaded_at >= start_date
        ).group_by(cast(FailedModel.uploaded_at, Date)).all()
        
        # Merge timelines into a single dict
        timeline_dict = defaultdict(lambda: {"date": None, "successful": 0, "failed": 0, "total": 0})
        
        for item in success_timeline:
            date_str = item.date.strftime('%Y-%m-%d')
            timeline_dict[date_str]["date"] = date_str
            timeline_dict[date_str]["successful"] = item.count
            timeline_dict[date_str]["total"] += item.count
        
        for item in failed_timeline:
            date_str = item.date.strftime('%Y-%m-%d')
            timeline_dict[date_str]["date"] = date_str
            timeline_dict[date_str]["failed"] = item.count
            timeline_dict[date_str]["total"] += item.count
        
        # Convert to sorted list
        timeline_data = sorted(timeline_dict.values(), key=lambda x: x["date"])
        
        # ============================================================
        # 3. FORMAT DISTRIBUTION (Successful invoices only)
        # ============================================================
        # Get all successful invoices within date range to extract actual formats
        success_invoices_for_format = db.query(SuccessModel).filter(
            SuccessModel.user_id == current_user.id,
            SuccessModel.deleted_at.is_(None),
            SuccessModel.uploaded_at >= start_date
        ).all()
        
        format_counts = defaultdict(int)
        
        for invoice in success_invoices_for_format:
            format_name = None
            
            # First, try to get from target_file_format if available
            if invoice.target_file_format:
                format_name = invoice.target_file_format
            else:
                # Extract from EDI file path/extension
                edi_path = invoice.blob_edi_path or invoice.edi_path
                
                if edi_path:
                    # Determine format from file extension
                    if edi_path.endswith('.x12') or edi_path.endswith('.810'):
                        format_name = 'X12'
                    elif edi_path.endswith('.edi') or edi_path.endswith('.edifact'):
                        format_name = 'EDIFACT'
                    elif edi_path.endswith('.xml'):
                        # Check if it's an embedded format
                        xml_path = invoice.blob_xml_path or invoice.xml_path
                        if xml_path:
                            # Try to detect embedded format from filename
                            if 'embed' in xml_path.lower():
                                format_name = 'XML_EMBED'
                            else:
                                format_name = 'XML'
                        else:
                            format_name = 'XML'
                    else:
                        # Check processing steps to determine format
                        if invoice.processing_steps:
                            try:
                                steps = invoice.processing_steps
                                for step in steps:
                                    if isinstance(step, dict):
                                        # Look for format hints in step messages
                                        message = step.get('message', '').lower()
                                        if 'x12' in message or '810' in message:
                                            format_name = 'X12'
                                            break
                                        elif 'edifact' in message:
                                            format_name = 'EDIFACT'
                                            break
                                        elif 'xml' in message and 'embed' in message:
                                            format_name = 'XML_EMBED'
                                            break
                            except:
                                pass
            
            # If still no format found, check if it has an EDI path (likely X12) or XML only
            if not format_name:
                edi_path = invoice.blob_edi_path or invoice.edi_path
                if edi_path:
                    format_name = 'X12'  # Most common EDI format
                else:
                    format_name = 'XML'  # Pure XML without conversion
            
            format_counts[format_name] += 1
        
        format_distribution = [
            {"format": format_name, "count": count}
            for format_name, count in format_counts.items()
        ]
        
        # Sort by count descending
        format_distribution = sorted(format_distribution, key=lambda x: x["count"], reverse=True)
        
        # ============================================================
        # 4. CUSTOMER DISTRIBUTION (Top 10)
        # ============================================================
        # Get successful and failed invoices to extract customer info
        success_invoices = db.query(SuccessModel).filter(
            SuccessModel.user_id == current_user.id,
            SuccessModel.deleted_at.is_(None)
        ).order_by(SuccessModel.uploaded_at.desc()).limit(500).all()  # Limit for performance
        
        failed_invoices_for_customers = db.query(FailedModel).filter(
            FailedModel.user_id == current_user.id,
            FailedModel.deleted_at.is_(None)
        ).order_by(FailedModel.uploaded_at.desc()).limit(500).all()  # Limit for performance
        
        customer_stats = defaultdict(lambda: {"successful": 0, "failed": 0, "name": None})
        
        # Process successful invoices
        for invoice in success_invoices:
            try:
                # Get XML path (blob or local)
                xml_path = invoice.blob_xml_path or invoice.xml_path
                
                if xml_path:
                    # Read XML content
                    xml_content = read_file_from_storage(xml_path)
                    
                    if xml_content:
                        # Extract customer info from XML
                        customer_id, customer_name = extract_supplier_info_from_string(xml_content)
                        
                        if customer_id and customer_name:
                            # Use customer name as key (more readable than ID)
                            customer_stats[customer_name]["successful"] += 1
                            customer_stats[customer_name]["name"] = customer_name
                        elif customer_id:
                            # Use ID if no name available
                            customer_stats[customer_id]["successful"] += 1
                            customer_stats[customer_id]["name"] = customer_id
            except Exception as e:
                logger.warning(f"⚠️ Failed to extract customer from invoice {invoice.id}: {e}")
                continue
        
        # Process failed invoices
        for invoice in failed_invoices_for_customers:
            try:
                # Get XML path (blob or local)
                xml_path = invoice.blob_xml_path or invoice.xml_path
                
                if xml_path:
                    # Read XML content
                    xml_content = read_file_from_storage(xml_path)
                    
                    if xml_content:
                        # Extract customer info from XML
                        customer_id, customer_name = extract_supplier_info_from_string(xml_content)
                        
                        if customer_id and customer_name:
                            customer_stats[customer_name]["failed"] += 1
                            customer_stats[customer_name]["name"] = customer_name
                        elif customer_id:
                            customer_stats[customer_id]["failed"] += 1
                            customer_stats[customer_id]["name"] = customer_id
            except Exception as e:
                logger.warning(f"⚠️ Failed to extract customer from failed invoice {invoice.id}: {e}")
                continue
        
        # Convert to list and filter out entries with no valid data
        customer_distribution = [
            {
                "customer": stats["name"] or customer_key,
                "successful": stats["successful"],
                "failed": stats["failed"],
                "total": stats["successful"] + stats["failed"]
            }
            for customer_key, stats in customer_stats.items()
            if stats["successful"] > 0 or stats["failed"] > 0  # Only include customers with invoices
        ]
        
        # Sort by total and get top 10
        customer_distribution = sorted(customer_distribution, key=lambda x: x["total"], reverse=True)[:10]
        
        # If no customer data found, provide helpful message
        if not customer_distribution:
            customer_distribution = [{
                "customer": "No Data Available",
                "successful": 0,
                "failed": 0,
                "total": 0
            }]
        
        # ============================================================
        # 5. REQUEST TYPE DISTRIBUTION (Web vs API)
        # ============================================================
        # Query successful invoices
        success_request_type_query = db.query(
            SuccessModel.request_type,
            func.count(SuccessModel.id).label('count')
        ).filter(
            SuccessModel.user_id == current_user.id,
            SuccessModel.deleted_at.is_(None)
        ).group_by(SuccessModel.request_type).all()
        
        # Query failed invoices
        failed_request_type_query = db.query(
            FailedModel.request_type,
            func.count(FailedModel.id).label('count')
        ).filter(
            FailedModel.user_id == current_user.id,
            FailedModel.deleted_at.is_(None)
        ).group_by(FailedModel.request_type).all()
        
        # Normalize and combine web/null entries from both success and failed
        type_counts = {"web": 0, "api": 0}
        
        for item in success_request_type_query:
            request_type = item.request_type
            # Normalize: treat null, empty, or 'web' as 'web'
            if not request_type or request_type.lower() == 'web':
                type_counts["web"] += item.count
            elif request_type.lower() == 'api':
                type_counts["api"] += item.count
        
        for item in failed_request_type_query:
            request_type = item.request_type
            # Normalize: treat null, empty, or 'web' as 'web'
            if not request_type or request_type.lower() == 'web':
                type_counts["web"] += item.count
            elif request_type.lower() == 'api':
                type_counts["api"] += item.count
        
        # Convert to list format, excluding zero counts
        request_type_distribution = [
            {"type": type_key, "count": count}
            for type_key, count in type_counts.items()
            if count > 0
        ]
        
        # ============================================================
        # 6. RECENT ACTIVITY (Last 10 invoices)
        # ============================================================
        recent_success = db.query(SuccessModel).filter(
            SuccessModel.user_id == current_user.id,
            SuccessModel.deleted_at.is_(None)
        ).order_by(SuccessModel.uploaded_at.desc()).limit(5).all()
        
        recent_failed = db.query(FailedModel).filter(
            FailedModel.user_id == current_user.id,
            FailedModel.deleted_at.is_(None)
        ).order_by(FailedModel.uploaded_at.desc()).limit(5).all()
        
        recent_activity = []
        
        for inv in recent_success:
            recent_activity.append({
                "id": inv.id,
                "tracking_id": str(inv.tracking_id),
                "status": "successful",
                "format": inv.target_file_format,
                "uploaded_at": inv.uploaded_at.isoformat() if inv.uploaded_at else None,
                "request_type": inv.request_type
            })
        
        for inv in recent_failed:
            recent_activity.append({
                "id": inv.id,
                "tracking_id": str(inv.tracking_id),
                "status": "failed",
                "format": inv.target_file_format,
                "uploaded_at": inv.uploaded_at.isoformat() if inv.uploaded_at else None,
                "request_type": inv.request_type
            })
        
        # Sort by date and limit to 10
        recent_activity = sorted(
            recent_activity,
            key=lambda x: x["uploaded_at"] or "",
            reverse=True
        )[:10]
        
        # ============================================================
        # RESPONSE
        # ============================================================
        logger.info(f"✅ Dashboard statistics calculated successfully")
        
        return {
            "overview": {
                "total": total_count,
                "successful": successful_count,
                "failed": failed_count,
                "success_rate": round(success_rate, 2)
            },
            "sat_overview": {
                "total": sat_total,
                "sent_to_sap": sat_sent_count,
                "pending": sat_pending_count
            },
            "timeline": timeline_data if timeline_data else [],
            "format_distribution": format_distribution if format_distribution else [],
            "customer_distribution": customer_distribution,
            "request_type_distribution": request_type_distribution if request_type_distribution else [],
            "recent_activity": recent_activity,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days
            },
            "has_data": total_count > 0 or sat_total > 0
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get dashboard statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dashboard statistics: {str(e)}"
        )


@router.get("/ai-insights")
async def get_ai_insights(
    current_user: ZodiacUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get AI-powered insights and analysis for failed invoices.
    Analyzes error patterns, identifies root causes, and provides actionable recommendations.
    """
    try:
        logger.info(f"🤖 Generating AI insights for user {current_user.id}")
        
        # Get recent failed invoices
        failed_invoices = db.query(FailedModel).filter(
            FailedModel.user_id == current_user.id,
            FailedModel.deleted_at.is_(None)
        ).order_by(FailedModel.uploaded_at.desc()).limit(20).all()
        
        if not failed_invoices:
            return {
                "summary": "🎉 No failed invoices found. Your processing is running smoothly!",
                "insights": [],
                "recommendations": [
                    "Continue monitoring your invoice processing for optimal performance.",
                    "Consider implementing automated validation checks before upload.",
                    "Review your success patterns to maintain high quality."
                ],
                "error_patterns": [],
                "root_causes": [],
                "total_failed": 0,
                "analyzed_at": datetime.utcnow().isoformat()
            }
        
        # ============================================================
        # ANALYZE ERROR PATTERNS
        # ============================================================
        error_patterns = defaultdict(int)
        error_details = []
        
        for invoice in failed_invoices:
            if invoice.processing_steps:
                for step in invoice.processing_steps:
                    if isinstance(step, dict):
                        if not step.get('success', True) and step.get('error_details'):
                            errors = step['error_details']
                            if isinstance(errors, list):
                                for error in errors:
                                    if isinstance(error, dict):
                                        error_code = error.get('error_code', 'UNKNOWN')
                                        error_patterns[error_code] += 1
                                        error_details.append({
                                            'code': error_code,
                                            'message': error.get('user_message', ''),
                                            'severity': error.get('severity', 'ERROR'),
                                            'tracking_id': str(invoice.tracking_id)
                                        })
        
        # Prepare context for AI
        total_failed = len(failed_invoices)
        top_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        
        error_summary = "\n".join([f"- {code}: {count} occurrences" for code, count in top_errors])
        
        sample_errors = error_details[:10]  # Take first 10 for AI analysis
        sample_errors_text = "\n".join([
            f"- [{e['severity']}] {e['code']}: {e['message']}"
            for e in sample_errors
        ])
        
        # ============================================================
        # AI ANALYSIS (if available)
        # ============================================================
        if OPENAI_API_KEY and openai_available:
            try:
                client = OpenAI(api_key=OPENAI_API_KEY)
                
                prompt = f"""
You are an expert EDI and e-invoice processing analyst. Analyze the following error patterns from {total_failed} failed invoice processing attempts and provide actionable insights.

Error Frequency:
{error_summary}

Sample Error Details:
{sample_errors_text}

Provide a JSON response with the following structure:
{{
    "summary": "Brief overview of the main issues (2-3 sentences)",
    "insights": [
        "Specific insight 1 about patterns or trends",
        "Specific insight 2 about common failure points",
        "Specific insight 3 about system behavior"
    ],
    "recommendations": [
        "Actionable recommendation 1",
        "Actionable recommendation 2",
        "Actionable recommendation 3"
    ],
    "root_causes": [
        "Identified root cause 1",
        "Identified root cause 2"
    ]
}}

Focus on:
1. Common patterns across errors
2. Root causes (XML structure, missing data, format issues, customer configuration)
3. Practical, actionable solutions
4. Prevention strategies
5. Priority of fixes (high impact first)

Keep insights concise, professional, and actionable. Respond ONLY with valid JSON, no markdown or extra text.
"""
                
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
                
                ai_response = json.loads(completion.choices[0].message.content)
                
                logger.info(f"✅ AI insights generated successfully")
                
                return {
                    "summary": ai_response.get("summary", "Analysis complete"),
                    "insights": ai_response.get("insights", []),
                    "recommendations": ai_response.get("recommendations", []),
                    "root_causes": ai_response.get("root_causes", []),
                    "error_patterns": [
                        {
                            "code": code,
                            "count": count,
                            "percentage": round(count / total_failed * 100, 1)
                        }
                        for code, count in top_errors
                    ],
                    "total_failed": total_failed,
                    "analyzed_at": datetime.utcnow().isoformat()
                }
            
            except Exception as ai_error:
                logger.warning(f"⚠️ AI analysis failed, using fallback: {ai_error}")
                # Continue to fallback
        
        # ============================================================
        # FALLBACK INSIGHTS (if AI not available or fails)
        # ============================================================
        return {
            "summary": f"Analyzed {total_failed} failed invoices. The most common issues are related to validation and format errors.",
            "insights": [
                f"Most frequent error: {top_errors[0][0]} ({top_errors[0][1]} occurrences)" if top_errors else "No specific error pattern detected",
                "XML validation errors are the primary cause of failures",
                "Review customer format configurations to ensure they match invoice types",
                "Consider enabling AI auto-correction to fix common issues automatically"
            ],
            "recommendations": [
                "✅ Validate XML files against UBL 2.1 schema before uploading",
                "✅ Ensure all required fields are populated (invoice ID, dates, parties)",
                "✅ Verify customer format configuration matches your invoice format",
                "✅ Use the AI correction feature to automatically fix common errors",
                "✅ Review failed invoice details to understand specific issues"
            ],
            "root_causes": [
                "XML schema validation failures",
                "Missing required fields or elements",
                "Format mismatch between invoice and customer configuration",
                "Invalid data formats (dates, amounts, identifiers)"
            ],
            "error_patterns": [
                {
                    "code": code,
                    "count": count,
                    "percentage": round(count / total_failed * 100, 1)
                }
                for code, count in top_errors
            ],
            "total_failed": total_failed,
            "analyzed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to generate AI insights: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate AI insights: {str(e)}"
        )


@router.get("/operations")
async def get_operations_statistics(
    days: int = Query(default=30, ge=1, le=365),
    current_user: ZodiacUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get operations statistics for the dashboard Operations tab:
    - Inbound/Outbound message counts
    - Auto-fix breakdown and savings
    - Processing time trends
    - External system status
    """
    try:
        logger.info(f"📊 Fetching operations statistics for last {days} days")
        
        # Calculate date range
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # ============================================================
        # 1. INBOUND/OUTBOUND MESSAGES
        # ============================================================
        # INBOUND: SAT Documents uploaded to send TO SAP
        # OUTBOUND: Invoices received FROM SAP and processed
        
        try:
            # Inbound: SAT Documents (files going TO SAP)
            # Use SATSimpleMerged which tracks sent_to_sap status
            inbound_total = db.query(func.count(SATSimpleMerged.id)).filter(
                SATSimpleMerged.user_id == current_user.id,
                SATSimpleMerged.created_at >= cutoff_date
            ).scalar() or 0
            
            # Count how many were successfully sent to SAP (sent_to_sap = True)
            inbound_successful = db.query(func.count(SATSimpleMerged.id)).filter(
                SATSimpleMerged.user_id == current_user.id,
                SATSimpleMerged.created_at >= cutoff_date,
                SATSimpleMerged.sent_to_sap == True
            ).scalar() or 0
            
            # Failed/pending inbound (not yet sent to SAP)
            inbound_failed = db.query(func.count(SATSimpleMerged.id)).filter(
                SATSimpleMerged.user_id == current_user.id,
                SATSimpleMerged.created_at >= cutoff_date,
                SATSimpleMerged.sent_to_sap == False
            ).scalar() or 0
            
            logger.info(f"📥 Inbound (SAT): {inbound_total} total, {inbound_successful} sent to SAP, {inbound_failed} pending")
        except Exception as sat_err:
            logger.warning(f"⚠️ Could not fetch SAT inbound stats: {sat_err}")
            inbound_total = 0
            inbound_successful = 0
            inbound_failed = 0
        
        # Outbound: Invoices (files FROM SAP being processed)
        outbound_successful = db.query(func.count(SuccessModel.id)).filter(
            SuccessModel.user_id == current_user.id,
            SuccessModel.uploaded_at >= cutoff_date
        ).scalar() or 0
        
        outbound_failed = db.query(func.count(FailedModel.id)).filter(
            FailedModel.user_id == current_user.id,
            FailedModel.uploaded_at >= cutoff_date
        ).scalar() or 0
        
        # ============================================================
        # 2. AUTO-FIX BREAKDOWN
        # ============================================================
        # Query correction cache for auto-fix statistics
        # Note: correction_cache stores corrections per customer, but we filter by user
        try:
            correction_stats = db.query(
                CorrectionCache.error_type,
                func.sum(CorrectionCache.success_count).label('total_fixes')
            ).filter(
                CorrectionCache.created_by_user_id == current_user.id,
                CorrectionCache.created_at >= cutoff_date,
                CorrectionCache.is_active == True
            ).group_by(CorrectionCache.error_type).all()
        except Exception as e:
            logger.warning(f"⚠️ Could not query correction_cache: {e}")
            correction_stats = []
        
        # Calculate total auto-fixes
        total_auto_fixes = sum(int(stat.total_fixes or 0) for stat in correction_stats) if correction_stats else 0
        
        # Map error types to user-friendly names and estimate time saved
        error_type_mapping = {
            'missing_invoice_id': {'name': 'Missing Fields', 'time_per_fix': 5},
            'missing_field': {'name': 'Missing Fields', 'time_per_fix': 5},
            'date_format': {'name': 'Date Format', 'time_per_fix': 3},
            'invalid_date': {'name': 'Date Format', 'time_per_fix': 3},
            'id_padding': {'name': 'ID Padding', 'time_per_fix': 2},
            'invalid_id_format': {'name': 'ID Padding', 'time_per_fix': 2},
            'party_info': {'name': 'Party Info', 'time_per_fix': 4},
            'missing_party': {'name': 'Party Info', 'time_per_fix': 4},
        }
        
        # Aggregate by user-friendly names
        auto_fix_breakdown = {}
        for stat in correction_stats:
            mapping = error_type_mapping.get(stat.error_type, {'name': 'Other', 'time_per_fix': 3})
            friendly_name = mapping['name']
            fix_count = int(stat.total_fixes or 0)
            time_saved = fix_count * mapping['time_per_fix']
            
            if friendly_name in auto_fix_breakdown:
                auto_fix_breakdown[friendly_name]['count'] += fix_count
                auto_fix_breakdown[friendly_name]['saved'] += time_saved
            else:
                auto_fix_breakdown[friendly_name] = {
                    'type': friendly_name,
                    'count': fix_count,
                    'saved': time_saved
                }
        
        # Convert to list and sort by count
        auto_fix_list = sorted(auto_fix_breakdown.values(), key=lambda x: x['count'], reverse=True) if auto_fix_breakdown else []
        
        # ============================================================
        # 3. PROCESSING TIME TREND (hourly averages)
        # ============================================================
        # Extract processing times from successful invoices
        processing_time_data = []
        
        # Get all successful invoices with processing_steps
        recent_invoices = db.query(SuccessModel).filter(
            SuccessModel.user_id == current_user.id,
            SuccessModel.uploaded_at >= cutoff_date,
            SuccessModel.processing_steps.isnot(None)
        ).all()
        
        # Aggregate by hour of day
        hourly_times = {}
        for invoice in recent_invoices:
            try:
                hour = invoice.uploaded_at.hour
                steps = json.loads(invoice.processing_steps) if isinstance(invoice.processing_steps, str) else invoice.processing_steps
                
                # Calculate total processing time
                total_time = sum(
                    step.get('duration_seconds', 0) 
                    for step in steps 
                    if isinstance(step, dict)
                )
                
                if hour not in hourly_times:
                    hourly_times[hour] = []
                hourly_times[hour].append(total_time)
            except:
                continue
        
        # Calculate averages
        for hour in range(24):
            times = hourly_times.get(hour, [])
            avg_time = round(sum(times) / len(times), 2) if times else 0
            processing_time_data.append({
                'hour': f'{hour:02d}:00',
                'avgTime': avg_time,
                'count': len(times)
            })
        
        # ============================================================
        # 4. EXTERNAL SYSTEM STATUS
        # ============================================================
        # Check external_status field (only exists in SuccessModel)
        external_success = db.query(func.count(SuccessModel.id)).filter(
            SuccessModel.user_id == current_user.id,
            SuccessModel.uploaded_at >= cutoff_date,
            SuccessModel.external_status == 'success'
        ).scalar() or 0
        
        external_failed = db.query(func.count(SuccessModel.id)).filter(
            SuccessModel.user_id == current_user.id,
            SuccessModel.uploaded_at >= cutoff_date,
            SuccessModel.external_status.in_(['failed', 'error'])
        ).scalar() or 0
        
        # All failed invoices are considered external failures
        # (since they failed before reaching external systems successfully)
        failed_invoice_count = db.query(func.count(FailedModel.id)).filter(
            FailedModel.user_id == current_user.id,
            FailedModel.uploaded_at >= cutoff_date
        ).scalar() or 0
        
        external_failed += failed_invoice_count
        
        external_pending = db.query(func.count(SuccessModel.id)).filter(
            SuccessModel.user_id == current_user.id,
            SuccessModel.uploaded_at >= cutoff_date,
            SuccessModel.external_status.in_(['pending', 'processing', 'False', None])
        ).scalar() or 0
        
        # ============================================================
        # RETURN RESPONSE
        # ============================================================
        has_operations_data = (
            (inbound_successful + inbound_failed) > 0 or 
            (outbound_successful + outbound_failed) > 0
        )
        
        return {
            "inbound": {
                "successful": inbound_successful,
                "failed": inbound_failed,
                "total": inbound_successful + inbound_failed
            },
            "outbound": {
                "successful": outbound_successful,
                "failed": outbound_failed,
                "total": outbound_successful + outbound_failed
            },
            "autoFix": {
                "total": total_auto_fixes,
                "successful": total_auto_fixes,  # Assume all cached corrections were successful
                "breakdown": auto_fix_list if auto_fix_list else []
            },
            "processingTime": {
                "hourly": processing_time_data if processing_time_data else [],
                "average": round(
                    sum(pt['avgTime'] for pt in processing_time_data) / len(processing_time_data), 2
                ) if processing_time_data else 0
            },
            "externalSystems": {
                "successful": external_success,
                "failed": external_failed,
                "pending": external_pending,
                "total": external_success + external_failed + external_pending
            },
            "has_data": has_operations_data
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch operations statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch operations statistics: {str(e)}"
        )


# ==================== Dashboard v2 endpoints ====================


@router.get("/v2/inbound")
async def get_dashboard_v2_inbound(
    days: int = Query(default=30, ge=1, le=365),
    current_user: ZodiacUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Dashboard v2 - Inbound process stats (SAT documents, merges, suppliers by RFC, tokens)."""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        total_documents = db.query(func.count(SATDocument.id)).filter(
            SATDocument.user_id == current_user.id,
            SATDocument.received_at >= cutoff_date
        ).scalar() or 0

        doc_type_rows = db.query(
            SATDocument.doc_type,
            func.count(SATDocument.id).label("count")
        ).filter(
            SATDocument.user_id == current_user.id,
            SATDocument.received_at >= cutoff_date
        ).group_by(SATDocument.doc_type).all()
        by_document_type = [{"doc_type": row.doc_type, "count": row.count} for row in doc_type_rows]

        source_rows = db.query(
            SATDocument.source,
            func.count(SATDocument.id).label("count")
        ).filter(
            SATDocument.user_id == current_user.id,
            SATDocument.received_at >= cutoff_date
        ).group_by(SATDocument.source).all()
        by_source = [{"source": row.source, "count": row.count} for row in source_rows]

        period_rows = db.query(
            SATDocument.fiscal_year,
            SATDocument.fiscal_period,
            func.count(SATDocument.id).label("count")
        ).filter(
            SATDocument.user_id == current_user.id,
            SATDocument.received_at >= cutoff_date,
            SATDocument.fiscal_year.isnot(None),
            SATDocument.fiscal_period.isnot(None)
        ).group_by(SATDocument.fiscal_year, SATDocument.fiscal_period).order_by(
            SATDocument.fiscal_year.desc(),
            SATDocument.fiscal_period.desc()
        ).limit(24).all()
        by_period = [
            {"fiscal_year": r.fiscal_year, "fiscal_period": r.fiscal_period, "count": r.count}
            for r in period_rows
        ]

        supplier_rows = db.query(
            SATDocument.supplier_rfc,
            SATDocument.supplier_name,
            func.count(SATDocument.id).label("count"),
            func.coalesce(
                func.sum(cast(func.nullif(func.trim(SATDocument.total), ""), Numeric(15, 2))),
                0
            ).label("total_amount")
        ).filter(
            SATDocument.user_id == current_user.id,
            SATDocument.received_at >= cutoff_date
        ).group_by(SATDocument.supplier_rfc, SATDocument.supplier_name).order_by(
            func.count(SATDocument.id).desc()
        ).limit(10).all()
        top_suppliers = [
            {
                "supplier_rfc": r.supplier_rfc,
                "supplier_name": r.supplier_name or r.supplier_rfc,
                "count": r.count,
                "total_amount": float(r.total_amount) if r.total_amount is not None else 0.0,
            }
            for r in supplier_rows
        ]

        merges_total = db.query(func.count(SATSimpleMerged.id)).filter(
            SATSimpleMerged.user_id == current_user.id,
            SATSimpleMerged.created_at >= cutoff_date
        ).scalar() or 0
        merges_sent = db.query(func.count(SATSimpleMerged.id)).filter(
            SATSimpleMerged.user_id == current_user.id,
            SATSimpleMerged.created_at >= cutoff_date,
            SATSimpleMerged.sent_to_sap == True
        ).scalar() or 0
        merges_pending = db.query(func.count(SATSimpleMerged.id)).filter(
            SATSimpleMerged.user_id == current_user.id,
            SATSimpleMerged.created_at >= cutoff_date,
            SATSimpleMerged.sent_to_sap == False
        ).scalar() or 0

        token_total = db.query(func.count(SupplierToken.id)).scalar() or 0
        token_active = db.query(func.count(SupplierToken.id)).filter(
            SupplierToken.is_active == True,
            (SupplierToken.expires_at.is_(None)) | (SupplierToken.expires_at >= datetime.utcnow())
        ).scalar() or 0
        token_expired = db.query(func.count(SupplierToken.id)).filter(
            SupplierToken.expires_at < datetime.utcnow(),
            SupplierToken.is_active == True
        ).scalar() or 0
        token_recently_used = db.query(func.count(SupplierToken.id)).filter(
            SupplierToken.last_used_at >= (datetime.utcnow() - timedelta(days=7))
        ).scalar() or 0

        return {
            "summary": {
                "total_documents": total_documents,
                "merges_total": merges_total,
                "merges_sent_to_sap": merges_sent,
                "merges_pending": merges_pending,
            },
            "by_document_type": by_document_type,
            "by_source": by_source,
            "by_period": by_period,
            "top_suppliers": top_suppliers,
            "tokens": {
                "total": token_total,
                "active": token_active,
                "expired": token_expired,
                "recently_used": token_recently_used,
            },
        }
    except Exception as e:
        logger.error(f"❌ Dashboard v2 inbound: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/v2/outbound")
async def get_dashboard_v2_outbound(
    days: int = Query(default=30, ge=1, le=365),
    current_user: ZodiacUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Dashboard v2 - Outbound process stats (V2 documents, validated, converted)."""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        documents_received = db.query(func.count(InvoiceV2Document.id)).filter(
            InvoiceV2Document.user_id == current_user.id,
            InvoiceV2Document.deleted_at.is_(None),
            InvoiceV2Document.uploaded_at >= cutoff_date
        ).scalar() or 0

        validated_query = db.query(
            InvoiceV2Validated.status,
            func.count(InvoiceV2Validated.id).label("count")
        ).join(InvoiceV2Document, InvoiceV2Validated.document_id == InvoiceV2Document.id).filter(
            InvoiceV2Document.user_id == current_user.id,
            InvoiceV2Document.uploaded_at >= cutoff_date
        ).group_by(InvoiceV2Validated.status).all()
        validated_success = sum(c for s, c in validated_query if s == "success")
        validated_failed = sum(c for s, c in validated_query if s == "failed")

        converted_query = db.query(
            ConvertedInvoice.conversion_status,
            func.count(ConvertedInvoice.id).label("count")
        ).join(InvoiceV2Validated, ConvertedInvoice.validated_invoice_id == InvoiceV2Validated.id).join(
            InvoiceV2Document, InvoiceV2Validated.document_id == InvoiceV2Document.id
        ).filter(
            InvoiceV2Document.user_id == current_user.id,
            InvoiceV2Document.uploaded_at >= cutoff_date
        ).group_by(ConvertedInvoice.conversion_status).all()
        converted_success = sum(c for s, c in converted_query if s == "success")
        converted_failed = sum(c for s, c in converted_query if s == "failed")
        converted_pending = sum(c for s, c in converted_query if s == "pending")
        converted_total = converted_success + converted_failed + converted_pending

        format_rows = db.query(
            ConvertedInvoice.target_format,
            func.count(ConvertedInvoice.id).label("count")
        ).join(InvoiceV2Validated, ConvertedInvoice.validated_invoice_id == InvoiceV2Validated.id).join(
            InvoiceV2Document, InvoiceV2Validated.document_id == InvoiceV2Document.id
        ).filter(
            InvoiceV2Document.user_id == current_user.id,
            InvoiceV2Document.uploaded_at >= cutoff_date,
            ConvertedInvoice.conversion_status == "success"
        ).group_by(ConvertedInvoice.target_format).all()
        by_format = [{"format": r.target_format, "count": r.count} for r in format_rows]

        customer_rows = db.query(
            ConvertedInvoice.customer_id,
            func.max(InvoiceV2Validated.invoice_data["customer_name"].astext).label("customer_name"),
            func.max(InvoiceV2Validated.invoice_data["currency"].astext).label("currency"),
            func.count(ConvertedInvoice.id).label("count"),
        ).join(InvoiceV2Validated, ConvertedInvoice.validated_invoice_id == InvoiceV2Validated.id).join(
            InvoiceV2Document, InvoiceV2Validated.document_id == InvoiceV2Document.id
        ).filter(
            InvoiceV2Document.user_id == current_user.id,
            InvoiceV2Document.uploaded_at >= cutoff_date
        ).filter(ConvertedInvoice.customer_id.isnot(None)).group_by(
            ConvertedInvoice.customer_id
        ).order_by(func.count(ConvertedInvoice.id).desc()).limit(10).all()
        top_customers = [
            {
                "customer_id": r.customer_id,
                "customer_name": r.customer_name or r.customer_id,
                "currency": r.currency or "—",
                "count": r.count,
            }
            for r in customer_rows
        ]

        timeline_docs = db.query(
            cast(InvoiceV2Document.uploaded_at, Date).label("date"),
            func.count(InvoiceV2Document.id).label("count")
        ).filter(
            InvoiceV2Document.user_id == current_user.id,
            InvoiceV2Document.deleted_at.is_(None),
            InvoiceV2Document.uploaded_at >= cutoff_date
        ).group_by(cast(InvoiceV2Document.uploaded_at, Date)).order_by(
            cast(InvoiceV2Document.uploaded_at, Date)
        ).all()
        timeline = [{"date": d.date.strftime("%Y-%m-%d") if d.date else None, "documents": d.count} for d in timeline_docs]

        validation_rate = (validated_success / (validated_success + validated_failed) * 100) if (validated_success + validated_failed) > 0 else 0
        conversion_rate = (converted_success / converted_total * 100) if converted_total > 0 else 0

        return {
            "funnel": {
                "documents_received": documents_received,
                "validated_success": validated_success,
                "validated_failed": validated_failed,
                "converted_success": converted_success,
                "converted_failed": converted_failed,
                "converted_pending": converted_pending,
                "converted_total": converted_total,
            },
            "summary": {
                "documents_received": documents_received,
                "validated_total": validated_success + validated_failed,
                "validated_success": validated_success,
                "validated_failed": validated_failed,
                "validation_success_rate_pct": round(validation_rate, 1),
                "converted_total": converted_total,
                "converted_success": converted_success,
                "converted_failed": converted_failed,
                "converted_pending": converted_pending,
                "conversion_success_rate_pct": round(conversion_rate, 1),
            },
            "by_format": by_format,
            "top_customers": top_customers,
            "timeline": timeline,
        }
    except Exception as e:
        logger.error(f"❌ Dashboard v2 outbound: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/v2/failed-invoices-analysis")
async def get_dashboard_v2_failed_invoices_analysis(
    days: int = Query(default=30, ge=1, le=365),
    demo: bool = Query(default=False, description="Return mock data for UI preview"),
    current_user: ZodiacUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Dashboard v2 - Failed invoices analytics: failure reasons, one-time vs repetitive, revenue loss."""
    if demo:
        return {
            "total_failed": 4,
            "failure_reasons": [
                {"reason": "missing:customer_name,tax_percentage", "count": 2},
                {"reason": "Invalid date format (issue_date)", "count": 1},
                {"reason": "Invalid currency code", "count": 1},
            ],
            "one_time_count": 2,
            "repetitive_count": 1,
            "repetitive_customer_ids": ["CUST-002"],
            "revenue_loss_by_currency": {"USD": 1250.5, "MXN": 15000.0},
            "by_date": [
                {"date": (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d"), "failed_count": 2, "revenue_loss": 500.0},
                {"date": (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d"), "failed_count": 2, "revenue_loss": 750.5},
            ],
        }
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        failed_list = (
            db.query(InvoiceV2Validated)
            .join(InvoiceV2Document, InvoiceV2Validated.document_id == InvoiceV2Document.id)
            .filter(
                InvoiceV2Document.user_id == current_user.id,
                InvoiceV2Document.deleted_at.is_(None),
                InvoiceV2Document.uploaded_at >= cutoff_date,
                InvoiceV2Validated.status == "failed",
            )
            .order_by(InvoiceV2Validated.validated_at.desc())
            .all()
        )
        total_failed = len(failed_list)

        failure_reasons_map = defaultdict(int)
        revenue_by_currency = defaultdict(float)
        customer_fail_counts = defaultdict(int)
        by_date_map = defaultdict(lambda: {"failed_count": 0, "revenue_loss": 0.0})

        for v in failed_list:
            reasons = []
            if v.missing_fields:
                key = "missing:" + ",".join(sorted(v.missing_fields))
                reasons.append(key)
            if v.validation_errors:
                for err in v.validation_errors if isinstance(v.validation_errors, list) else []:
                    if isinstance(err, dict):
                        msg = err.get("message", "") or err.get("field", "error")
                        reasons.append((msg[:50] + "..") if len(msg) > 50 else msg)
                    else:
                        reasons.append("validation_error")
            if not reasons:
                reasons.append("unknown")
            for r in reasons:
                failure_reasons_map[r] += 1

            inv = v.invoice_data or {}
            try:
                total_val = inv.get("total") or inv.get("payable_amount")
                if total_val is not None:
                    amt = float(total_val) if not isinstance(total_val, (int, float)) else float(total_val)
                    cur = (inv.get("currency") or "USD").strip() or "USD"
                    revenue_by_currency[cur] += amt
            except (TypeError, ValueError):
                pass

            cid = (inv.get("customer_id") or "").strip() or "unknown"
            customer_fail_counts[cid] += 1

            vdate = v.validated_at.date() if v.validated_at else None
            if vdate:
                by_date_map[vdate.strftime("%Y-%m-%d")]["failed_count"] += 1
                try:
                    total_val = inv.get("total") or inv.get("payable_amount")
                    if total_val is not None:
                        amt = float(total_val) if not isinstance(total_val, (int, float)) else float(total_val)
                        by_date_map[vdate.strftime("%Y-%m-%d")]["revenue_loss"] += amt
                except (TypeError, ValueError):
                    pass

        one_time_count = sum(1 for c in customer_fail_counts.values() if c == 1)
        repetitive_count = sum(1 for c in customer_fail_counts.values() if c > 1)
        repetitive_customer_ids = [cid for cid, count in customer_fail_counts.items() if count > 1 and cid != "unknown"]

        failure_reasons = [{"reason": r, "count": c} for r, c in sorted(failure_reasons_map.items(), key=lambda x: -x[1])]
        revenue_loss_by_currency = dict(revenue_by_currency)
        by_date = [
            {"date": d, "failed_count": by_date_map[d]["failed_count"], "revenue_loss": round(by_date_map[d]["revenue_loss"], 2)}
            for d in sorted(by_date_map.keys())
        ]

        return {
            "total_failed": total_failed,
            "failure_reasons": failure_reasons,
            "one_time_count": one_time_count,
            "repetitive_count": repetitive_count,
            "repetitive_customer_ids": repetitive_customer_ids,
            "revenue_loss_by_currency": revenue_loss_by_currency,
            "by_date": by_date,
        }
    except Exception as e:
        logger.error(f"❌ Dashboard v2 failed-invoices-analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/v2/failed-invoices-ai-insights")
async def get_dashboard_v2_failed_invoices_ai_insights(
    days: int = Query(default=30, ge=1, le=365),
    demo: bool = Query(default=False, description="Return mock AI insights for UI preview"),
    current_user: ZodiacUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Dashboard v2 - AI insights for failed invoices (summary, insights, recommendations, root causes)."""
    if demo:
        return {
            "summary": "Demo: 4 failed invoices in the period. Main issues are missing required fields (customer_name, tax_percentage), invalid date format, and invalid currency. One customer has repetitive failures.",
            "insights": [
                "Missing customer_name and tax_percentage is the most frequent failure reason (2 invoices).",
                "One customer (CUST-002) has multiple failures; consider a dedicated review for this customer.",
                "Revenue at risk: USD 1,250.50 and MXN 15,000.00 across failed validations.",
            ],
            "recommendations": [
                "Ensure UBL invoices include AccountingCustomerParty/cac:Party/cac:PartyName/cbc:Name and tax percentage where required.",
                "Validate date formats (issue_date, due_date) against ISO 8601 before submission.",
                "Review currency codes against ISO 4217; fix invalid or empty currency in source systems.",
            ],
            "root_causes": [
                "Incomplete or blank required fields in supplier XML export.",
                "Date/currency format mismatches between ERP and validation rules.",
            ],
            "revenue_impact_note": "Total revenue at risk: USD 1,250.50, MXN 15,000.00.",
            "analyzed_at": datetime.utcnow().isoformat(),
        }
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        failed_list = (
            db.query(InvoiceV2Validated)
            .join(InvoiceV2Document, InvoiceV2Validated.document_id == InvoiceV2Document.id)
            .filter(
                InvoiceV2Document.user_id == current_user.id,
                InvoiceV2Document.deleted_at.is_(None),
                InvoiceV2Document.uploaded_at >= cutoff_date,
                InvoiceV2Validated.status == "failed",
            )
            .order_by(InvoiceV2Validated.validated_at.desc())
            .limit(50)
            .all()
        )
        if not failed_list:
            return {
                "summary": "No failed invoices in this period.",
                "insights": [],
                "recommendations": [],
                "root_causes": [],
                "revenue_impact_note": None,
                "analyzed_at": datetime.utcnow().isoformat(),
            }
        failure_reasons_map = defaultdict(int)
        revenue_by_currency = defaultdict(float)
        customer_fail_counts = defaultdict(int)
        examples = []
        for v in failed_list[:10]:
            inv = v.invoice_data or {}
            reasons = []
            if v.missing_fields:
                reasons.append("missing: " + ", ".join(v.missing_fields))
            if v.validation_errors and isinstance(v.validation_errors, list):
                for err in v.validation_errors:
                    if isinstance(err, dict):
                        reasons.append(err.get("message", err.get("field", "error")))
            if not reasons:
                reasons.append("unknown")
            examples.append({
                "customer_id": inv.get("customer_id", "—"),
                "customer_name": inv.get("customer_name", "—"),
                "total": inv.get("total") or inv.get("payable_amount"),
                "currency": inv.get("currency", "—"),
                "reasons": reasons,
            })
            if v.missing_fields:
                key = "missing:" + ",".join(sorted(v.missing_fields))
                failure_reasons_map[key] += 1
            if v.validation_errors and isinstance(v.validation_errors, list):
                for err in v.validation_errors:
                    if isinstance(err, dict):
                        msg = (err.get("message") or err.get("field") or "error")[:80]
                        failure_reasons_map[msg] += 1
            try:
                total_val = inv.get("total") or inv.get("payable_amount")
                if total_val is not None:
                    amt = float(total_val) if not isinstance(total_val, (int, float)) else float(total_val)
                    cur = (inv.get("currency") or "USD").strip() or "USD"
                    revenue_by_currency[cur] += amt
            except (TypeError, ValueError):
                pass
            cid = (inv.get("customer_id") or "").strip() or "unknown"
            customer_fail_counts[cid] += 1
        one_time = sum(1 for c in customer_fail_counts.values() if c == 1)
        repetitive = sum(1 for c in customer_fail_counts.values() if c > 1)
        top_reasons = sorted(failure_reasons_map.items(), key=lambda x: -x[1])[:8]
        reason_text = "\n".join([f"- {r}: {c} occurrences" for r, c in top_reasons])
        revenue_text = ", ".join([f"{cur} {amt:.2f}" for cur, amt in revenue_by_currency.items()])
        examples_text = "\n".join([
            f"Customer {ex['customer_id']} ({ex['customer_name']}), total {ex['currency']} {ex['total']}: " + "; ".join(ex["reasons"])
            for ex in examples[:3]
        ])
        context = f"""
Failed invoices in period: {len(failed_list)} (showing up to 50).

Failure reason counts:
{reason_text}

Revenue at risk by currency: {revenue_text or 'None'}

One-time failures (customers with 1 failure): {one_time}. Repetitive (customers with >1 failure): {repetitive}.

Example failed invoices and their reasons:
{examples_text}
"""
        if OPENAI_API_KEY and openai_available:
            try:
                client = OpenAI(api_key=OPENAI_API_KEY)
                prompt = f"""
You are an expert e-invoice and UBL analyst. Analyze the following failed invoice validation data and provide actionable insights.

{context}

Provide a JSON response with this exact structure:
{{
    "summary": "Brief 2-3 sentence overview of the main issues",
    "insights": ["Specific insight 1", "Specific insight 2", "Specific insight 3"],
    "recommendations": ["Actionable recommendation 1", "Actionable recommendation 2", "Actionable recommendation 3"],
    "root_causes": ["Root cause 1", "Root cause 2"],
    "revenue_impact_note": "One sentence on revenue at risk by currency, or null if none"
}}

Focus on: common patterns, missing/required fields, format issues, repetitive vs one-time, and practical fixes.
Respond ONLY with valid JSON, no markdown or extra text.
"""
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    response_format={"type": "json_object"},
                )
                ai_response = json.loads(completion.choices[0].message.content)
                return {
                    "summary": ai_response.get("summary", "Analysis complete."),
                    "insights": ai_response.get("insights", []),
                    "recommendations": ai_response.get("recommendations", []),
                    "root_causes": ai_response.get("root_causes", []),
                    "revenue_impact_note": ai_response.get("revenue_impact_note"),
                    "analyzed_at": datetime.utcnow().isoformat(),
                }
            except Exception as ai_err:
                logger.warning(f"V2 failed-invoices AI insights OpenAI error: {ai_err}")
        top_3 = top_reasons[:3]
        return {
            "summary": f"Analyzed {len(failed_list)} failed invoices. Top issues: " + "; ".join([f"{r} ({c})" for r, c in top_3]) + ".",
            "insights": [
                f"Most frequent: {top_3[0][0]} ({top_3[0][1]} occurrences)" if top_3 else "No patterns",
                "Review missing_fields and validation_errors in the Failed tab for details.",
                f"Revenue at risk: {revenue_text}" if revenue_text else "No revenue totals in failed records.",
            ],
            "recommendations": [
                "Fix missing required fields in source XML (see failure_reasons).",
                "Validate date and currency formats before upload.",
                "For repetitive failures by customer, check customer-specific configuration.",
            ],
            "root_causes": [
                "Missing or invalid required UBL fields.",
                "Format validation failures (dates, amounts, codes).",
            ],
            "revenue_impact_note": f"Total at risk: {revenue_text}" if revenue_text else None,
            "analyzed_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Dashboard v2 failed-invoices-ai-insights: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/v2/business")
async def get_dashboard_v2_business(
    days: int = Query(default=90, ge=1, le=365),
    currency: str = Query(default=None, description="Filter by currency code (e.g. NZD, USD). Omit for all."),
    current_user: ZodiacUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Dashboard v2 - Business: products by industry, industry breakdown, trend. Data comes from successful invoices' line_items."""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        rows = db.query(InvoiceV2BusinessData).filter(
            InvoiceV2BusinessData.user_id == current_user.id,
            InvoiceV2BusinessData.created_at >= cutoff_date,
            InvoiceV2BusinessData.products.isnot(None),
            InvoiceV2BusinessData.industry.isnot(None),
        ).all()

        # If no BI data yet but user has successful validated invoices, backfill from line_items
        if not rows:
            success_count = db.query(InvoiceV2Validated).join(
                InvoiceV2Document,
                InvoiceV2Validated.document_id == InvoiceV2Document.id
            ).filter(
                InvoiceV2Document.user_id == current_user.id,
                InvoiceV2Validated.status == "success"
            ).count()
            if success_count > 0:
                _run_backfill_invoice_v2_bi(db, current_user, max_invoices=500)
                rows = db.query(InvoiceV2BusinessData).filter(
                    InvoiceV2BusinessData.user_id == current_user.id,
                    InvoiceV2BusinessData.created_at >= cutoff_date,
                    InvoiceV2BusinessData.products.isnot(None),
                    InvoiceV2BusinessData.industry.isnot(None),
                ).all()

        # Revenue by currency (from full dataset, before currency filter)
        currency_revenue = defaultdict(lambda: {"count": 0, "revenue": Decimal("0")})
        for r in rows:
            curr = (r.currency or "Unknown").strip() or "Unknown"
            currency_revenue[curr]["count"] += 1
            currency_revenue[curr]["revenue"] += (r.total_amount or Decimal("0"))
        revenue_by_currency = [
            {
                "currency": curr,
                "invoice_count": int(data["count"]),
                "total_revenue": float(data["revenue"]),
            }
            for curr, data in sorted(currency_revenue.items(), key=lambda x: -x[1]["revenue"])
        ][:15]

        # Currency filter (optional): restrict to selected currency
        if currency and str(currency).strip():
            curr_upper = str(currency).strip().upper()
            rows = [r for r in rows if (r.currency or "").strip().upper() == curr_upper]

        product_industry = defaultdict(lambda: {"count": 0, "revenue": Decimal("0"), "quantity": Decimal("0"), "unit_counts": defaultdict(int)})
        industry_totals = defaultdict(lambda: {"count": 0, "revenue": Decimal("0")})

        for r in rows:
            industry = r.industry or "General"
            amount = (r.total_amount or Decimal("0"))
            products = r.products if isinstance(r.products, list) else []
            industry_totals[industry]["count"] += 1
            industry_totals[industry]["revenue"] += amount
            for p in products:
                if not isinstance(p, dict):
                    continue
                name = (p.get("name") or p.get("item_name") or "Unknown").strip() or "Unknown"
                key = (name, industry)
                product_industry[key]["count"] += 1
                unit = (p.get("unit_code") or "").strip() or None
                if unit:
                    product_industry[key]["unit_counts"][unit] += 1
                qty = p.get("quantity")
                if qty is not None:
                    try:
                        product_industry[key]["quantity"] += Decimal(str(qty))
                    except Exception:
                        pass
                line_revenue = p.get("revenue") or p.get("line_extension_amount")
                if line_revenue is not None:
                    try:
                        product_industry[key]["revenue"] += Decimal(str(line_revenue))
                    except Exception:
                        product_industry[key]["revenue"] += amount / len(products) if products else amount
                else:
                    product_industry[key]["revenue"] += amount / len(products) if products else amount

        def _most_common_unit(unit_counts):
            if not unit_counts:
                return None
            return max(unit_counts.items(), key=lambda x: x[1])[0]

        products_by_industry = [
            {
                "product_name": name,
                "industry": ind,
                "unit_of_measure": _most_common_unit(data["unit_counts"]) or "—",
                "invoice_count": data["count"],
                "revenue": float(data["revenue"]),
            }
            for (name, ind), data in sorted(product_industry.items(), key=lambda x: -x[1]["revenue"])
        ][:100]

        # Quantity & price analysis: total units sold, avg price per product+industry (for scatter chart)
        quantity_price_analysis = []
        for (name, ind), data in product_industry.items():
            total_qty = float(data["quantity"])
            rev = float(data["revenue"])
            avg_price = (rev / total_qty) if total_qty and total_qty > 0 else None
            quantity_price_analysis.append({
                "product_name": name,
                "industry": ind,
                "unit_of_measure": _most_common_unit(data["unit_counts"]) or "—",
                "total_quantity": round(total_qty, 2),
                "avg_price": round(avg_price, 2) if avg_price is not None else None,
                "revenue": rev,
            })
        quantity_price_analysis = sorted(
            [x for x in quantity_price_analysis if x["total_quantity"] > 0],
            key=lambda x: -x["total_quantity"]
        )[:50]

        industry_breakdown = [
            {"industry": ind, "count": data["count"], "total_revenue": float(data["revenue"])}
            for ind, data in sorted(industry_totals.items(), key=lambda x: -x[1]["revenue"])
        ]

        # Revenue by customer (customer_id = RFC from successful data)
        customer_revenue = defaultdict(lambda: {"customer_name": None, "count": 0, "revenue": Decimal("0")})
        for r in rows:
            cid = r.customer_id or "Unknown"
            customer_revenue[cid]["customer_name"] = r.customer_name or cid
            customer_revenue[cid]["count"] += 1
            customer_revenue[cid]["revenue"] += (r.total_amount or Decimal("0"))
        revenue_by_customer = [
            {
                "customer_id": cid,
                "customer_name": data["customer_name"] or cid,
                "invoice_count": data["count"],
                "total_revenue": float(data["revenue"]),
            }
            for cid, data in sorted(customer_revenue.items(), key=lambda x: -x[1]["revenue"])
        ][:50]

        # Revenue by country (customer_country from successful invoices)
        country_revenue = defaultdict(lambda: {"count": 0, "revenue": Decimal("0")})
        for r in rows:
            country = r.customer_country or "Unknown"
            country_revenue[country]["count"] += 1
            country_revenue[country]["revenue"] += (r.total_amount or Decimal("0"))
        revenue_by_country = [
            {
                "country": country,
                "country_name": _country_code_to_name(country),
                "invoice_count": int(data["count"]),
                "total_revenue": float(data["revenue"]),
            }
            for country, data in sorted(country_revenue.items(), key=lambda x: -x[1]["revenue"])
        ][:20]

        # Customers by country (distinct customers per country - histogram)
        country_customers = defaultdict(set)  # country -> set of customer_id
        for r in rows:
            country = r.customer_country or "Unknown"
            cust_key = r.customer_id or r.customer_name or f"anon_{r.id}"
            country_customers[country].add(cust_key)
        customers_by_country = [
            {
                "country": country,
                "country_name": _country_code_to_name(country),
                "customer_count": len(cust_set),
            }
            for country, cust_set in sorted(country_customers.items(), key=lambda x: -len(x[1]))
        ][:20]

        mid = cutoff_date + (datetime.utcnow() - cutoff_date) / 2
        prev_cutoff = cutoff_date - (datetime.utcnow() - cutoff_date)
        current_revenue = db.query(func.coalesce(func.sum(InvoiceV2BusinessData.total_amount), 0)).filter(
            InvoiceV2BusinessData.user_id == current_user.id,
            InvoiceV2BusinessData.created_at >= mid,
        ).scalar() or 0
        previous_revenue = db.query(func.coalesce(func.sum(InvoiceV2BusinessData.total_amount), 0)).filter(
            InvoiceV2BusinessData.user_id == current_user.id,
            InvoiceV2BusinessData.created_at >= prev_cutoff,
            InvoiceV2BusinessData.created_at < mid,
        ).scalar() or 0
        try:
            current_revenue = float(current_revenue)
            previous_revenue = float(previous_revenue)
        except Exception:
            current_revenue = previous_revenue = 0.0
        trend_pct = ((current_revenue - previous_revenue) / previous_revenue * 100) if previous_revenue else 0.0

        trend = {
            "current_period_revenue": current_revenue,
            "previous_period_revenue": previous_revenue,
            "revenue_change_pct": round(trend_pct, 1),
        }

        # AI insights about the whole business tab (using existing API key)
        ai_insights = None
        if OPENAI_API_KEY and openai_available and rows:
            try:
                client = OpenAI(api_key=OPENAI_API_KEY)
                prompt = f"""You are a business analyst. Based on the following dashboard data (from validated invoices), provide concise AI insights in JSON format.

REVENUE TREND:
- Current period revenue: {current_revenue}
- Previous period revenue: {previous_revenue}
- Change: {trend_pct:+.1f}%

INDUSTRY BREAKDOWN (top 10):
{json.dumps(industry_breakdown[:10], indent=2)}

REVENUE BY CUSTOMER (top 10):
{json.dumps(revenue_by_customer[:10], indent=2)}

REVENUE BY COUNTRY (top 10):
{json.dumps(revenue_by_country[:10], indent=2)}

REVENUE BY CURRENCY:
{json.dumps(revenue_by_currency, indent=2)}

CUSTOMERS BY COUNTRY (top 10):
{json.dumps(customers_by_country[:10], indent=2)}

PRODUCTS BY INDUSTRY (top 15):
{json.dumps(products_by_industry[:15], indent=2)}

Respond with a single JSON object with this structure (no markdown, only valid JSON):
{{
  "summary": "2-3 sentence overall summary of the business situation",
  "revenue_insights": ["insight about revenue trend"],
  "industry_insights": ["insight about industry mix"],
  "customer_insights": ["insight about top customers / concentration"],
  "country_insights": ["insight about geographic distribution / top countries"],
  "currency_insights": ["insight about revenue by currency / multi-currency mix"],
  "product_insights": ["insight about product performance"],
  "recommendations": ["1-3 actionable recommendations"]
}}"""

                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    response_format={"type": "json_object"},
                )
                ai_insights = json.loads(completion.choices[0].message.content)
                logger.info("AI business insights generated for v2/business")
            except Exception as ai_err:
                logger.warning(f"AI insights for v2/business failed: {ai_err}")
                ai_insights = None

        return {
            "products_by_industry": products_by_industry,
            "industry_breakdown": industry_breakdown,
            "quantity_price_analysis": quantity_price_analysis,
            "revenue_by_customer": revenue_by_customer,
            "revenue_by_country": revenue_by_country,
            "revenue_by_currency": revenue_by_currency,
            "customers_by_country": customers_by_country,
            "trend": trend,
            "ai_insights": ai_insights,
        }
    except Exception as e:
        logger.error(f"❌ Dashboard v2 business: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/v2/customer-comparison")
async def get_dashboard_v2_customer_comparison(
    days: int = Query(default=90, ge=1, le=365),
    currency: str = Query(default=None, description="Filter by currency code. Omit for all."),
    current_user: ZodiacUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Customer comparison: per-customer product breakdown and revenue. For interactive customer vs customer analysis."""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        rows = db.query(InvoiceV2BusinessData).filter(
            InvoiceV2BusinessData.user_id == current_user.id,
            InvoiceV2BusinessData.created_at >= cutoff_date,
            InvoiceV2BusinessData.products.isnot(None),
        ).all()

        if not rows:
            success_count = db.query(InvoiceV2Validated).join(
                InvoiceV2Document,
                InvoiceV2Validated.document_id == InvoiceV2Document.id
            ).filter(
                InvoiceV2Document.user_id == current_user.id,
                InvoiceV2Validated.status == "success"
            ).count()
            if success_count > 0:
                _run_backfill_invoice_v2_bi(db, current_user, max_invoices=500)
                rows = db.query(InvoiceV2BusinessData).filter(
                    InvoiceV2BusinessData.user_id == current_user.id,
                    InvoiceV2BusinessData.created_at >= cutoff_date,
                    InvoiceV2BusinessData.products.isnot(None),
                ).all()

        # Currency filter
        if currency and str(currency).strip():
            curr_upper = str(currency).strip().upper()
            rows = [r for r in rows if (r.currency or "").strip().upper() == curr_upper]

        # Revenue by currency (for filter dropdown, from full dataset)
        all_rows = db.query(InvoiceV2BusinessData).filter(
            InvoiceV2BusinessData.user_id == current_user.id,
            InvoiceV2BusinessData.created_at >= cutoff_date,
            InvoiceV2BusinessData.products.isnot(None),
        ).all()
        curr_rev = defaultdict(lambda: {"count": 0, "revenue": Decimal("0")})
        for r in all_rows:
            c = (r.currency or "Unknown").strip() or "Unknown"
            curr_rev[c]["count"] += 1
            curr_rev[c]["revenue"] += (r.total_amount or Decimal("0"))
        revenue_by_currency = [
            {"currency": c, "invoice_count": d["count"], "total_revenue": float(d["revenue"])}
            for c, d in sorted(curr_rev.items(), key=lambda x: -x[1]["revenue"])
        ]

        # Aggregate: customer -> { total_revenue, invoice_count, industry_counts, currency_revenue, products }
        cust_data = defaultdict(lambda: {
            "customer_id": None,
            "customer_name": None,
            "customer_country": None,
            "invoice_count": 0,
            "total_revenue": Decimal("0"),
            "industry_counts": defaultdict(int),
            "currency_revenue": defaultdict(lambda: Decimal("0")),
            "products": defaultdict(lambda: {"revenue": Decimal("0"), "quantity": Decimal("0"), "unit": None}),
        })
        for r in rows:
            cid = r.customer_id or r.customer_name or f"anon_{r.id}"
            cust_data[cid]["customer_id"] = r.customer_id
            cust_data[cid]["customer_name"] = r.customer_name or cid
            cust_data[cid]["customer_country"] = r.customer_country
            cust_data[cid]["invoice_count"] += 1
            cust_data[cid]["total_revenue"] += (r.total_amount or Decimal("0"))
            if r.industry:
                cust_data[cid]["industry_counts"][r.industry] += 1
            curr = (r.currency or "Unknown").strip() or "Unknown"
            cust_data[cid]["currency_revenue"][curr] += (r.total_amount or Decimal("0"))
            products = r.products if isinstance(r.products, list) else []
            for p in products:
                if not isinstance(p, dict):
                    continue
                name = (p.get("name") or p.get("item_name") or "Unknown").strip() or "Unknown"
                rev = p.get("revenue") or p.get("line_extension_amount")
                qty = p.get("quantity")
                unit = (p.get("unit_code") or "").strip() or None
                try:
                    if rev is not None:
                        cust_data[cid]["products"][name]["revenue"] += Decimal(str(rev))
                    if qty is not None:
                        cust_data[cid]["products"][name]["quantity"] += Decimal(str(qty))
                    if unit:
                        cust_data[cid]["products"][name]["unit"] = unit
                except Exception:
                    pass

        # Revenue by country (for map)
        country_revenue = defaultdict(lambda: {"count": 0, "revenue": Decimal("0")})
        for r in rows:
            c = (r.customer_country or "Unknown").strip() or "Unknown"
            country_revenue[c]["count"] += 1
            country_revenue[c]["revenue"] += (r.total_amount or Decimal("0"))
        revenue_by_country = [
            {"country": c, "country_name": _country_code_to_name(c), "invoice_count": d["count"], "total_revenue": float(d["revenue"])}
            for c, d in sorted(country_revenue.items(), key=lambda x: -float(x[1]["revenue"]))
        ][:30]

        customers = []
        for cid, data in sorted(cust_data.items(), key=lambda x: -float(x[1]["total_revenue"])):
            products_list = [
                {
                    "product_name": pname,
                    "revenue": round(float(pdata["revenue"]), 2),
                    "quantity": round(float(pdata["quantity"]), 2),
                    "unit": pdata["unit"] or "—",
                }
                for pname, pdata in sorted(data["products"].items(), key=lambda y: -float(y[1]["revenue"]))
            ]
            industry = max(data["industry_counts"].items(), key=lambda x: x[1])[0] if data["industry_counts"] else None
            currencies = [{"currency": c, "revenue": round(float(rev), 2)} for c, rev in sorted(data["currency_revenue"].items(), key=lambda x: -float(x[1]))]
            customers.append({
                "customer_id": data["customer_id"],
                "customer_name": data["customer_name"],
                "customer_country": data["customer_country"],
                "industry": industry,
                "currencies": currencies,
                "invoice_count": data["invoice_count"],
                "total_revenue": round(float(data["total_revenue"]), 2),
                "products": products_list,
            })

        return {
            "customers": customers,
            "revenue_by_currency": revenue_by_currency,
            "revenue_by_country": revenue_by_country,
        }
    except Exception as e:
        logger.error(f"❌ Dashboard v2 customer-comparison: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/v2/customer-comparison-chat")
async def post_customer_comparison_chat(
    message: str = Body(..., embed=True),
    customer_a: dict = Body(..., embed=True),
    customer_b: dict = Body(..., embed=True),
    conversation_history: list = Body(default=[], embed=True),
    current_user: ZodiacUser = Depends(get_current_user),
):
    """AI chat for customer comparison: initial summary and follow-up Q&A about two customers."""
    if not OPENAI_API_KEY or not openai_available:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="AI chat not available (OpenAI key missing)")
    try:
        sys_content = f"""You are a business analyst assistant. The user is comparing two customers. Use ONLY the data below to answer. Be concise and factual.

CUSTOMER A:
- Name: {customer_a.get('customer_name', 'N/A')}
- Country: {customer_a.get('customer_country', 'N/A')}
- Industry: {customer_a.get('industry', 'N/A')}
- Total Revenue: {customer_a.get('total_revenue', 0)}
- Invoices: {customer_a.get('invoice_count', 0)}
- Currencies: {json.dumps(customer_a.get('currencies', []))}
- Top products: {json.dumps((customer_a.get('products') or [])[:5])}

CUSTOMER B:
- Name: {customer_b.get('customer_name', 'N/A')}
- Country: {customer_b.get('customer_country', 'N/A')}
- Industry: {customer_b.get('industry', 'N/A')}
- Total Revenue: {customer_b.get('total_revenue', 0)}
- Invoices: {customer_b.get('invoice_count', 0)}
- Currencies: {json.dumps(customer_b.get('currencies', []))}
- Top products: {json.dumps((customer_b.get('products') or [])[:5])}

Answer the user's question based only on this data. If asked for a summary first, provide 2-3 sentences comparing revenue, geography, industry, and product mix."""
        messages = [{"role": "system", "content": sys_content}]
        for h in conversation_history[-10:]:
            if isinstance(h, dict) and h.get("role") and h.get("content"):
                messages.append({"role": h["role"], "content": str(h["content"])[:2000]})
        messages.append({"role": "user", "content": message[:1500]})
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4,
            max_tokens=500,
        )
        reply = (resp.choices[0].message.content or "").strip()
        return {"reply": reply}
    except Exception as e:
        logger.warning(f"Customer comparison chat failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


def _get_ai_analysis_config():
    """
    Use same env var names as invoice-bot (config.example) for the AI analysis page only.
    Enables a single set of env vars (e.g. OPENAI_API_KEY) for both invoice-bot and this page.
    """
    openai_key = os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
    return openai_key


def _get_sales_by_product_from_vbrp(db: Session, limit: int = 10) -> list[dict]:
    """
    Highest sales by product from vbrp, enriched with:
    - product_name (from MAKT.MAKTX if available)
    - total_quantity (sum of FKIMG or similar)
    - unit_of_measure (VRKME / MEINS)
    - currency (from VBRK.WAERK if available)
    """
    bind = db.get_bind()
    if bind is None:
        return []

    inspector = inspect(bind)
    table_names = inspector.get_table_names()
    table_map = {name.lower(): name for name in table_names}

    vbrp_table = table_map.get("vbrp")
    if not vbrp_table:
        return []

    vbrp_cols = inspector.get_columns(vbrp_table)
    vbrp_map = {c["name"].lower(): c["name"] for c in vbrp_cols}

    product_candidates = ["matnr", "product_id", "material"]
    value_candidates = ["netwr", "net_value", "amount", "sales_value"]
    qty_candidates = ["fkimg", "quantity", "qty"]
    uom_candidates = ["vrkme", "meins", "unit"]

    product_col = next((vbrp_map[c] for c in product_candidates if c in vbrp_map), None)
    value_col = next((vbrp_map[c] for c in value_candidates if c in vbrp_map), None)
    qty_col = next((vbrp_map[c] for c in qty_candidates if c in vbrp_map), None)
    uom_col = next((vbrp_map[c] for c in uom_candidates if c in vbrp_map), None)

    if not product_col or not value_col:
        return []

    # Optional product description (MAKT)
    makt_table = table_map.get("makt")
    makt_matnr = makt_maktx = None
    if makt_table:
        makt_cols = inspector.get_columns(makt_table)
        makt_map = {c["name"].lower(): c["name"] for c in makt_cols}
        makt_matnr = makt_map.get("matnr")
        makt_maktx = makt_map.get("maktx")

    # Optional currency from billing header (VBRK)
    vbrk_table = table_map.get("vbrk")
    vbrk_vbeln = vbrk_waerk = None
    if vbrk_table:
        vbrk_cols = inspector.get_columns(vbrk_table)
        vbrk_map = {c["name"].lower(): c["name"] for c in vbrk_cols}
        vbrk_vbeln = vbrk_map.get("vbeln")
        vbrk_waerk = vbrk_map.get("waerk")

    vbeln_vbrp = vbrp_map.get("vbeln")

    select_parts = [f"vbrp.{product_col} AS product_id"]
    group_by_parts = [f"vbrp.{product_col}"]

    if makt_table and makt_matnr and makt_maktx and vbrp_map.get("matnr"):
        select_parts.append(f"m.{makt_maktx} AS product_name")
        group_by_parts.append(f"m.{makt_maktx}")

    if qty_col:
        select_parts.append(f"SUM(CAST(vbrp.{qty_col} AS NUMERIC)) AS total_quantity")
    if uom_col:
        select_parts.append(f"vbrp.{uom_col} AS unit_of_measure")
        group_by_parts.append(f"vbrp.{uom_col}")

    if vbrk_table and vbrk_vbeln and vbrk_waerk and vbeln_vbrp:
        select_parts.append(f"vbrk.{vbrk_waerk} AS currency")
        group_by_parts.append(f"vbrk.{vbrk_waerk}")

    select_parts.append(f"SUM(CAST(vbrp.{value_col} AS NUMERIC)) AS total_sales")

    join_makt = ""
    if makt_table and makt_matnr and vbrp_map.get("matnr"):
        join_makt = f'LEFT JOIN "{makt_table}" m ON vbrp.{vbrp_map["matnr"]} = m.{makt_matnr}'

    join_vbrk = ""
    if vbrk_table and vbrk_vbeln and vbeln_vbrp:
        join_vbrk = f'LEFT JOIN "{vbrk_table}" vbrk ON vbrp.{vbeln_vbrp} = vbrk.{vbrk_vbeln}'

    sql = f"""
        SELECT
            {", ".join(select_parts)}
        FROM "{vbrp_table}" vbrp
        {join_makt}
        {join_vbrk}
        GROUP BY {", ".join(group_by_parts)}
        ORDER BY total_sales DESC
        LIMIT :limit
    """

    rows = db.execute(text(sql), {"limit": limit}).fetchall()

    results: list[dict] = []
    for r in rows:
        total = r.total_sales
        if isinstance(total, Decimal):
            total = float(total)
        total_qty = getattr(r, "total_quantity", None)
        if isinstance(total_qty, Decimal):
            total_qty = float(total_qty)
        results.append(
            {
                "product_id": str(getattr(r, "product_id", "")),
                "product_name": str(getattr(r, "product_name", "")) if hasattr(r, "product_name") else None,
                "total_sales": float(total) if total is not None else 0.0,
                "total_quantity": float(total_qty) if total_qty is not None else None,
                "unit_of_measure": str(getattr(r, "unit_of_measure", "")) if hasattr(r, "unit_of_measure") else None,
                "currency": str(getattr(r, "currency", "")) if hasattr(r, "currency") else None,
            }
        )

    return results


def _get_sales_vs_invoice_v2(db: Session) -> dict:
    """
    Compare total sales from vbrp with total revenue from InvoiceV2BusinessData.
    Used so AI can talk about differences between raw SAP billing data and Zodiac invoice BI data.
    """
    bind = db.get_bind()
    if bind is None:
        return {}

    inspector = inspect(bind)
    table_names = inspector.get_table_names()
    table_map = {name.lower(): name for name in table_names}

    vbrp_table = table_map.get("vbrp")
    if not vbrp_table:
        return {}

    cols = inspector.get_columns(vbrp_table)
    name_map = {c["name"].lower(): c["name"] for c in cols}
    value_candidates = ["netwr", "net_value", "amount", "sales_value"]
    value_col = next((name_map[c] for c in value_candidates if c in name_map), None)
    if not value_col:
        return {}

    # Sum from vbrp
    q_vbrp = text(f"SELECT SUM(CAST({value_col} AS NUMERIC)) AS total_sales FROM {vbrp_table}")
    row = db.execute(q_vbrp).fetchone()
    sales_total = row.total_sales if row and row.total_sales is not None else 0
    if isinstance(sales_total, Decimal):
        sales_total = float(sales_total)

    # Sum from InvoiceV2BusinessData (Zodiac BI)
    inv_total = db.query(func.coalesce(func.sum(InvoiceV2BusinessData.total_amount), 0)).scalar() or 0
    try:
        inv_total = float(inv_total)
    except Exception:
        inv_total = 0.0

    diff = sales_total - inv_total
    return {
        "sales_total_from_vbrp": sales_total,
        "revenue_total_from_invoice_v2": inv_total,
        "difference": diff,
    }


def _get_lowest_sales_by_customer_country(db: Session, limit: int = 10) -> list[dict]:
    """
    Aggregate lowest sales by customer and country using vbrp + vbrk + kna1 if present.
    """
    bind = db.get_bind()
    if bind is None:
        return []

    inspector = inspect(bind)
    tables = {t.lower(): t for t in inspector.get_table_names()}
    vbrp_table = tables.get("vbrp")
    vbrk_table = tables.get("vbrk")
    kna1_table = tables.get("kna1")
    if not vbrp_table or not vbrk_table:
        return []

    # Column maps (case-insensitive)
    vbrp_cols = {c["name"].lower(): c["name"] for c in inspector.get_columns(vbrp_table)}
    vbrk_cols = {c["name"].lower(): c["name"] for c in inspector.get_columns(vbrk_table)}
    kna1_cols = {c["name"].lower(): c["name"] for c in inspector.get_columns(kna1_table)} if kna1_table else {}

    # Join keys and fields
    vbeln_vbrp = vbrp_cols.get("vbeln")
    vbeln_vbrk = vbrk_cols.get("vbeln")
    kunag = vbrk_cols.get("kunag")
    if not vbeln_vbrp or not vbeln_vbrk or not kunag:
        return []

    value_candidates = ["netwr", "net_value", "amount", "sales_value"]
    value_col = next((vbrp_cols[c] for c in value_candidates if c in vbrp_cols), None)
    if not value_col:
        return []

    kna1_kunnr = kna1_cols.get("kunnr")
    kna1_name1 = kna1_cols.get("name1") or kna1_cols.get("name")
    kna1_land1 = kna1_cols.get("land1") or kna1_cols.get("country")

    customer_expr = f"COALESCE(k.{kna1_name1}, vbrk.{kunag})" if kna1_table and kna1_name1 else f"vbrk.{kunag}"
    country_expr = f"COALESCE(k.{kna1_land1}, 'Unknown')" if kna1_table and kna1_land1 else "'Unknown'"

    join_kna1 = ""
    if kna1_table and kna1_kunnr:
        # Quote the table name so Postgres uses the actual mixed-case identifier (e.g. "KNA1")
        join_kna1 = f'LEFT JOIN "{kna1_table}" k ON vbrk.{kunag} = k.{kna1_kunnr}'

    sql = f"""
        SELECT
            {customer_expr} AS customer_name,
            {country_expr} AS country,
            SUM(CAST(vbrp.{value_col} AS NUMERIC)) AS total_sales
        FROM "{vbrp_table}" vbrp
        JOIN "{vbrk_table}" vbrk ON vbrp.{vbeln_vbrp} = vbrk.{vbeln_vbrk}
        {join_kna1}
        GROUP BY {customer_expr}, {country_expr}
        ORDER BY total_sales ASC
        LIMIT :limit
    """
    rows = db.execute(text(sql), {"limit": limit}).fetchall()

    results: list[dict] = []
    for r in rows:
        total = r.total_sales
        if isinstance(total, Decimal):
            total = float(total)
        results.append(
            {
                "customer_name": str(r.customer_name) if r.customer_name is not None else "Unknown",
                "country": str(r.country) if r.country is not None else "Unknown",
                "total_sales": float(total) if total is not None else 0.0,
            }
        )
    return results


def _get_sales_by_country_industry(db: Session, limit: int = 20) -> list[dict]:
    """
    Aggregate sales by country and industry using vbrp + vbrk + kna1 (BRSCH) if present.
    """
    bind = db.get_bind()
    if bind is None:
        return []

    inspector = inspect(bind)
    tables = {t.lower(): t for t in inspector.get_table_names()}
    vbrp_table = tables.get("vbrp")
    vbrk_table = tables.get("vbrk")
    kna1_table = tables.get("kna1")
    if not vbrp_table or not vbrk_table or not kna1_table:
        return []

    vbrp_cols = {c["name"].lower(): c["name"] for c in inspector.get_columns(vbrp_table)}
    vbrk_cols = {c["name"].lower(): c["name"] for c in inspector.get_columns(vbrk_table)}
    kna1_cols = {c["name"].lower(): c["name"] for c in inspector.get_columns(kna1_table)}

    vbeln_vbrp = vbrp_cols.get("vbeln")
    vbeln_vbrk = vbrk_cols.get("vbeln")
    kunag = vbrk_cols.get("kunag")
    if not vbeln_vbrp or not vbeln_vbrk or not kunag:
        return []

    value_candidates = ["netwr", "net_value", "amount", "sales_value"]
    value_col = next((vbrp_cols[c] for c in value_candidates if c in vbrp_cols), None)
    if not value_col:
        return []

    kna1_kunnr = kna1_cols.get("kunnr")
    kna1_land1 = kna1_cols.get("land1") or kna1_cols.get("country")
    kna1_brsch = kna1_cols.get("brsch") or kna1_cols.get("industry")
    if not kna1_kunnr or not kna1_land1 or not kna1_brsch:
        return []

    sql = f"""
        SELECT
            k.{kna1_land1} AS country,
            k.{kna1_brsch} AS industry,
            SUM(CAST(vbrp.{value_col} AS NUMERIC)) AS total_sales,
            COUNT(DISTINCT vbrk.{kunag}) AS customer_count
        FROM "{vbrp_table}" vbrp
        JOIN "{vbrk_table}" vbrk ON vbrp.{vbeln_vbrp} = vbrk.{vbeln_vbrk}
        JOIN "{kna1_table}" k ON vbrk.{kunag} = k.{kna1_kunnr}
        GROUP BY k.{kna1_land1}, k.{kna1_brsch}
        ORDER BY total_sales DESC
        LIMIT :limit
    """
    rows = db.execute(text(sql), {"limit": limit}).fetchall()

    results: list[dict] = []
    for r in rows:
        total = r.total_sales
        if isinstance(total, Decimal):
            total = float(total)
        results.append(
            {
                "country": str(r.country) if r.country is not None else "Unknown",
                "industry": str(r.industry) if r.industry is not None else "Unknown",
                "total_sales": float(total) if total is not None else 0.0,
                "customer_count": int(r.customer_count or 0),
            }
        )
    return results


def _get_sales_by_customer_product_country(db: Session, limit: int = 10) -> list[dict]:
    """
    Aggregate highest sales by customer, product, and country using vbrp + vbrk + kna1.
    """
    bind = db.get_bind()
    if bind is None:
        return []

    inspector = inspect(bind)
    tables = {t.lower(): t for t in inspector.get_table_names()}
    vbrp_table = tables.get("vbrp")
    vbrk_table = tables.get("vbrk")
    kna1_table = tables.get("kna1")
    if not vbrp_table or not vbrk_table:
        return []

    vbrp_cols = {c["name"].lower(): c["name"] for c in inspector.get_columns(vbrp_table)}
    vbrk_cols = {c["name"].lower(): c["name"] for c in inspector.get_columns(vbrk_table)}
    kna1_cols = {c["name"].lower(): c["name"] for c in inspector.get_columns(kna1_table)} if kna1_table else {}

    vbeln_vbrp = vbrp_cols.get("vbeln")
    vbeln_vbrk = vbrk_cols.get("vbeln")
    kunag = vbrk_cols.get("kunag")
    if not vbeln_vbrp or not vbeln_vbrk or not kunag:
        return []

    product_candidates = ["matnr", "product_id", "material"]
    value_candidates = ["netwr", "net_value", "amount", "sales_value"]
    product_col = next((vbrp_cols[c] for c in product_candidates if c in vbrp_cols), None)
    value_col = next((vbrp_cols[c] for c in value_candidates if c in vbrp_cols), None)
    if not product_col or not value_col:
        return []

    kna1_kunnr = kna1_cols.get("kunnr")
    kna1_name1 = kna1_cols.get("name1") or kna1_cols.get("name")
    kna1_land1 = kna1_cols.get("land1") or kna1_cols.get("country")

    customer_expr = f"COALESCE(k.{kna1_name1}, vbrk.{kunag})" if kna1_table and kna1_name1 else f"vbrk.{kunag}"
    country_expr = f"COALESCE(k.{kna1_land1}, 'Unknown')" if kna1_table and kna1_land1 else "'Unknown'"

    join_kna1 = ""
    if kna1_table and kna1_kunnr:
        # Quote the table name so Postgres uses the actual mixed-case identifier (e.g. "KNA1")
        join_kna1 = f'LEFT JOIN "{kna1_table}" k ON vbrk.{kunag} = k.{kna1_kunnr}'

    sql = f"""
        SELECT
            {customer_expr} AS customer_name,
            {country_expr} AS country,
            vbrp.{product_col} AS product_id,
            SUM(CAST(vbrp.{value_col} AS NUMERIC)) AS total_sales
        FROM "{vbrp_table}" vbrp
        JOIN "{vbrk_table}" vbrk ON vbrp.{vbeln_vbrp} = vbrk.{vbeln_vbrk}
        {join_kna1}
        GROUP BY {customer_expr}, {country_expr}, vbrp.{product_col}
        ORDER BY total_sales DESC
        LIMIT :limit
    """
    rows = db.execute(text(sql), {"limit": limit}).fetchall()

    results: list[dict] = []
    for r in rows:
        total = r.total_sales
        if isinstance(total, Decimal):
            total = float(total)
        results.append(
            {
                "customer_name": str(r.customer_name) if r.customer_name is not None else "Unknown",
                "country": str(r.country) if r.country is not None else "Unknown",
                "product_id": str(r.product_id),
                "total_sales": float(total) if total is not None else 0.0,
            }
        )
    return results


def _build_ai_analysis_context(context_keys: list, current_user: ZodiacUser, db: Session, days: int = 30) -> str:
    """Build context string for AI analysis chat from requested context_keys.
    Uses V2 pipeline (InvoiceV2Document, InvoiceV2Validated, ConvertedInvoice) for outbound;
    SATDocument, SATSimpleMerged for inbound. Context keys: stats, failed_summary, top_customers,
    inbound_summary, business_summary, process_flow.
    
    Also includes table availability and date ranges from cached schemas.
    """
    if not context_keys:
        return ""
    parts = []
    cutoff_date = datetime.utcnow() - timedelta(days=days)

    if "stats" in context_keys:
        try:
            documents_received = db.query(func.count(InvoiceV2Document.id)).filter(
                InvoiceV2Document.user_id == current_user.id,
                InvoiceV2Document.deleted_at.is_(None),
                InvoiceV2Document.uploaded_at >= cutoff_date,
            ).scalar() or 0
            validated_query = db.query(
                InvoiceV2Validated.status,
                func.count(InvoiceV2Validated.id).label("count"),
            ).join(InvoiceV2Document, InvoiceV2Validated.document_id == InvoiceV2Document.id).filter(
                InvoiceV2Document.user_id == current_user.id,
                InvoiceV2Document.uploaded_at >= cutoff_date,
            ).group_by(InvoiceV2Validated.status).all()
            validated_success = sum(c for s, c in validated_query if s == "success")
            validated_failed = sum(c for s, c in validated_query if s == "failed")
            converted_query = db.query(
                ConvertedInvoice.conversion_status,
                func.count(ConvertedInvoice.id).label("count"),
            ).join(InvoiceV2Validated, ConvertedInvoice.validated_invoice_id == InvoiceV2Validated.id).join(
                InvoiceV2Document, InvoiceV2Validated.document_id == InvoiceV2Document.id,
            ).filter(
                InvoiceV2Document.user_id == current_user.id,
                InvoiceV2Document.uploaded_at >= cutoff_date,
            ).group_by(ConvertedInvoice.conversion_status).all()
            converted_success = sum(c for s, c in converted_query if s == "success")
            converted_failed = sum(c for s, c in converted_query if s == "failed")
            converted_pending = sum(c for s, c in converted_query if s == "pending")
            converted_total = converted_success + converted_failed + converted_pending
            validation_rate = (
                (validated_success / (validated_success + validated_failed) * 100)
                if (validated_success + validated_failed) > 0 else 0
            )
            conversion_rate = (converted_success / converted_total * 100) if converted_total > 0 else 0
        except Exception:
            documents_received = validated_success = validated_failed = 0
            converted_success = converted_failed = converted_pending = converted_total = 0
            validation_rate = conversion_rate = 0
        try:
            sat_total = db.query(func.count(SATSimpleMerged.id)).filter(
                SATSimpleMerged.user_id == current_user.id,
                SATSimpleMerged.created_at >= cutoff_date,
            ).scalar() or 0
            sat_sent = db.query(func.count(SATSimpleMerged.id)).filter(
                SATSimpleMerged.user_id == current_user.id,
                SATSimpleMerged.created_at >= cutoff_date,
                SATSimpleMerged.sent_to_sap == True,
            ).scalar() or 0
            sat_pending = db.query(func.count(SATSimpleMerged.id)).filter(
                SATSimpleMerged.user_id == current_user.id,
                SATSimpleMerged.created_at >= cutoff_date,
                SATSimpleMerged.sent_to_sap == False,
            ).scalar() or 0
        except Exception:
            sat_total = sat_sent = sat_pending = 0
        parts.append(
            f"Dashboard statistics (last {days} days). "
            f"Outbound (V2): {documents_received} documents received; "
            f"validated: {validated_success} success, {validated_failed} failed (rate {validation_rate:.1f}%); "
            f"converted: {converted_success} success, {converted_failed} failed, {converted_pending} pending (rate {conversion_rate:.1f}%). "
            f"Inbound (SAT): {sat_total} merged documents ({sat_sent} sent to SAP, {sat_pending} pending)."
        )

    if "failed_summary" in context_keys:
        try:
            failed_count_v2 = (
                db.query(InvoiceV2Validated)
                .join(InvoiceV2Document, InvoiceV2Validated.document_id == InvoiceV2Document.id)
                .filter(
                    InvoiceV2Document.user_id == current_user.id,
                    InvoiceV2Document.deleted_at.is_(None),
                    InvoiceV2Document.uploaded_at >= cutoff_date,
                    InvoiceV2Validated.status == "failed",
                )
                .count()
            )
            parts.append(
                f"Failed invoices (V2 validations, last {days} days): {failed_count_v2} failed."
            )
        except Exception:
            parts.append("Failed invoices (V2): data unavailable.")

    if "top_customers" in context_keys:
        try:
            customer_rows = db.query(
                ConvertedInvoice.customer_id,
                func.max(InvoiceV2Validated.invoice_data["customer_name"].astext).label("customer_name"),
                func.max(InvoiceV2Validated.invoice_data["currency"].astext).label("currency"),
                func.count(ConvertedInvoice.id).label("count"),
            ).join(InvoiceV2Validated, ConvertedInvoice.validated_invoice_id == InvoiceV2Validated.id).join(
                InvoiceV2Document, InvoiceV2Validated.document_id == InvoiceV2Document.id,
            ).filter(
                InvoiceV2Document.user_id == current_user.id,
                InvoiceV2Document.uploaded_at >= cutoff_date,
            ).filter(ConvertedInvoice.customer_id.isnot(None)).group_by(
                ConvertedInvoice.customer_id,
            ).order_by(func.count(ConvertedInvoice.id).desc()).limit(10).all()
            top_customers_list = [
                {
                    "customer_id": r.customer_id,
                    "customer_name": (r.customer_name or r.customer_id) or "—",
                    "currency": (r.currency or "—").strip() or "—",
                    "invoice_count": r.count,
                }
                for r in customer_rows
            ]
            if top_customers_list:
                parts.append("Top customers (outbound, by invoice count): " + json.dumps(top_customers_list))
            else:
                parts.append("Top customers (outbound): no customer data in the period.")
        except Exception as e:
            logger.warning(f"AI context top_customers: {e}")
            parts.append("Top customers (outbound): data unavailable.")

    if "inbound_summary" in context_keys:
        try:
            total_documents = db.query(func.count(SATDocument.id)).filter(
                SATDocument.user_id == current_user.id,
                SATDocument.received_at >= cutoff_date,
            ).scalar() or 0
            merges_total = db.query(func.count(SATSimpleMerged.id)).filter(
                SATSimpleMerged.user_id == current_user.id,
                SATSimpleMerged.created_at >= cutoff_date,
            ).scalar() or 0
            merges_sent = db.query(func.count(SATSimpleMerged.id)).filter(
                SATSimpleMerged.user_id == current_user.id,
                SATSimpleMerged.created_at >= cutoff_date,
                SATSimpleMerged.sent_to_sap == True,
            ).scalar() or 0
            merges_pending = db.query(func.count(SATSimpleMerged.id)).filter(
                SATSimpleMerged.user_id == current_user.id,
                SATSimpleMerged.created_at >= cutoff_date,
                SATSimpleMerged.sent_to_sap == False,
            ).scalar() or 0
            supplier_rows = db.query(
                SATDocument.supplier_rfc,
                SATDocument.supplier_name,
                func.count(SATDocument.id).label("count"),
                func.coalesce(
                    func.sum(cast(func.nullif(func.trim(SATDocument.total), ""), Numeric(15, 2))),
                    0,
                ).label("total_amount"),
            ).filter(
                SATDocument.user_id == current_user.id,
                SATDocument.received_at >= cutoff_date,
            ).group_by(SATDocument.supplier_rfc, SATDocument.supplier_name).order_by(
                func.count(SATDocument.id).desc(),
            ).limit(10).all()
            top_suppliers = [
                {
                    "supplier_rfc": r.supplier_rfc,
                    "supplier_name": (r.supplier_name or r.supplier_rfc) or "—",
                    "count": r.count,
                    "total_amount": float(r.total_amount) if r.total_amount is not None else 0.0,
                }
                for r in supplier_rows
            ]
            parts.append(
                f"Inbound (SAT): {total_documents} documents received; "
                f"{merges_total} merges ({merges_sent} sent to SAP, {merges_pending} pending). "
                f"Top suppliers: " + json.dumps(top_suppliers),
            )
        except Exception as e:
            logger.warning(f"AI context inbound_summary: {e}")
            parts.append("Inbound (SAT): data unavailable.")

    if "business_summary" in context_keys:
        try:
            rows = db.query(InvoiceV2BusinessData).filter(
                InvoiceV2BusinessData.user_id == current_user.id,
                InvoiceV2BusinessData.created_at >= cutoff_date,
            ).all()
            customer_revenue = defaultdict(lambda: {"customer_name": None, "count": 0, "revenue": Decimal("0")})
            country_revenue = defaultdict(lambda: {"count": 0, "revenue": Decimal("0")})
            for r in rows:
                cid = r.customer_id or "Unknown"
                customer_revenue[cid]["customer_name"] = r.customer_name or cid
                customer_revenue[cid]["count"] += 1
                customer_revenue[cid]["revenue"] += (r.total_amount or Decimal("0"))
                country = r.customer_country or "Unknown"
                country_revenue[country]["count"] += 1
                country_revenue[country]["revenue"] += (r.total_amount or Decimal("0"))
            revenue_by_customer = sorted(
                [
                    {"customer_id": cid, "customer_name": d["customer_name"] or cid, "invoice_count": d["count"], "total_revenue": float(d["revenue"])}
                    for cid, d in customer_revenue.items()
                ],
                key=lambda x: -x["total_revenue"],
            )[:10]
            revenue_by_country = sorted(
                [
                    {"country": c, "invoice_count": d["count"], "total_revenue": float(d["revenue"])}
                    for c, d in country_revenue.items()
                ],
                key=lambda x: -x["total_revenue"],
            )[:5]
            mid = cutoff_date + (datetime.utcnow() - cutoff_date) / 2
            prev_cutoff = cutoff_date - (datetime.utcnow() - cutoff_date)
            current_revenue = db.query(func.coalesce(func.sum(InvoiceV2BusinessData.total_amount), 0)).filter(
                InvoiceV2BusinessData.user_id == current_user.id,
                InvoiceV2BusinessData.created_at >= mid,
            ).scalar() or 0
            previous_revenue = db.query(func.coalesce(func.sum(InvoiceV2BusinessData.total_amount), 0)).filter(
                InvoiceV2BusinessData.user_id == current_user.id,
                InvoiceV2BusinessData.created_at >= prev_cutoff,
                InvoiceV2BusinessData.created_at < mid,
            ).scalar() or 0
            try:
                current_revenue = float(current_revenue)
                previous_revenue = float(previous_revenue)
            except Exception:
                current_revenue = previous_revenue = 0.0
            trend_pct = ((current_revenue - previous_revenue) / previous_revenue * 100) if previous_revenue else 0.0
            # Optional: derive sales by product and related breakdowns directly from vbrp in the primary DB
            product_sales: list[dict] = []
            lowest_sales_by_customer_country: list[dict] = []
            sales_by_country_industry: list[dict] = []
            sales_by_customer_product_country: list[dict] = []
            sales_vs_invoice = {}
            try:
                product_sales = _get_sales_by_product_from_vbrp(db, limit=10)
            except Exception as e2:
                logger.warning(f"AI context product_sales from vbrp: {e2}")
            try:
                lowest_sales_by_customer_country = _get_lowest_sales_by_customer_country(db, limit=10)
            except Exception as e2:
                logger.warning(f"AI context lowest_sales_by_customer_country: {e2}")
            try:
                sales_by_country_industry = _get_sales_by_country_industry(db, limit=20)
            except Exception as e2:
                logger.warning(f"AI context sales_by_country_industry: {e2}")
            try:
                sales_by_customer_product_country = _get_sales_by_customer_product_country(db, limit=10)
            except Exception as e2:
                logger.warning(f"AI context sales_by_customer_product_country: {e2}")
            try:
                sales_vs_invoice = _get_sales_vs_invoice_v2(db)
            except Exception as e2:
                logger.warning(f"AI context sales_vs_invoice_v2: {e2}")

            parts.append(
                f"Business (revenue): current period {current_revenue:.0f}, previous {previous_revenue:.0f} (change {trend_pct:+.1f}%). "
                f"Revenue by customer (top 10): {json.dumps(revenue_by_customer)}. "
                f"Revenue by country (top 5): {json.dumps(revenue_by_country)}. "
                f"Sales by product (top 10 from vbrp): {json.dumps(product_sales)}. "
                f"Lowest sales by customer and country (from vbrp/vbrk/kna1): {json.dumps(lowest_sales_by_customer_country)}. "
                f"Sales by country and industry (from vbrp/vbrk/kna1): {json.dumps(sales_by_country_industry)}. "
                f"Highest sales by customer, product, and country (from vbrp/vbrk/kna1): {json.dumps(sales_by_customer_product_country)}. "
                f"Comparison of total sales (vbrp) vs InvoiceV2BusinessData: {json.dumps(sales_vs_invoice)}.",
            )
        except Exception as e:
            logger.warning(f"AI context business_summary: {e}")
            parts.append("Business summary: data unavailable.")

    if "process_flow" in context_keys:
        try:
            documents_received = db.query(func.count(InvoiceV2Document.id)).filter(
                InvoiceV2Document.user_id == current_user.id,
                InvoiceV2Document.deleted_at.is_(None),
                InvoiceV2Document.uploaded_at >= cutoff_date,
            ).scalar() or 0
            validated_query = db.query(
                InvoiceV2Validated.status,
                func.count(InvoiceV2Validated.id).label("count"),
            ).join(InvoiceV2Document, InvoiceV2Validated.document_id == InvoiceV2Document.id).filter(
                InvoiceV2Document.user_id == current_user.id,
                InvoiceV2Document.uploaded_at >= cutoff_date,
            ).group_by(InvoiceV2Validated.status).all()
            validated_success = sum(c for s, c in validated_query if s == "success")
            validated_failed = sum(c for s, c in validated_query if s == "failed")
            converted_query = db.query(
                ConvertedInvoice.conversion_status,
                func.count(ConvertedInvoice.id).label("count"),
            ).join(InvoiceV2Validated, ConvertedInvoice.validated_invoice_id == InvoiceV2Validated.id).join(
                InvoiceV2Document, InvoiceV2Validated.document_id == InvoiceV2Document.id,
            ).filter(
                InvoiceV2Document.user_id == current_user.id,
                InvoiceV2Document.uploaded_at >= cutoff_date,
            ).group_by(ConvertedInvoice.conversion_status).all()
            converted_success = sum(c for s, c in converted_query if s == "success")
            converted_failed = sum(c for s, c in converted_query if s == "failed")
            converted_pending = sum(c for s, c in converted_query if s == "pending")
            merges_total = db.query(func.count(SATSimpleMerged.id)).filter(
                SATSimpleMerged.user_id == current_user.id,
                SATSimpleMerged.created_at >= cutoff_date,
            ).scalar() or 0
            merges_sent = db.query(func.count(SATSimpleMerged.id)).filter(
                SATSimpleMerged.user_id == current_user.id,
                SATSimpleMerged.created_at >= cutoff_date,
                SATSimpleMerged.sent_to_sap == True,
            ).scalar() or 0
            merges_pending = db.query(func.count(SATSimpleMerged.id)).filter(
                SATSimpleMerged.user_id == current_user.id,
                SATSimpleMerged.created_at >= cutoff_date,
                SATSimpleMerged.sent_to_sap == False,
            ).scalar() or 0
            parts.append(
                "Standard process flows. Outbound: Document received -> Validation (success/fail) -> Conversion (format, success/fail/pending) -> Output. "
                f"Current outbound counts: received {documents_received}, validated success {validated_success} failed {validated_failed}, "
                f"converted success {converted_success} failed {converted_failed} pending {converted_pending}. "
                "Inbound: SAT documents received -> Merge (batch) -> Send to SAP. "
                f"Current inbound: {merges_total} merges, {merges_sent} sent to SAP, {merges_pending} pending.",
            )
        except Exception as e:
            logger.warning(f"AI context process_flow: {e}")
            parts.append(
                "Standard process flows. Outbound: Document received -> Validation -> Conversion -> Output. "
                "Inbound: SAT documents -> Merge -> Send to SAP. Current counts unavailable.",
            )
    
    # Add table availability and date ranges from cached schemas
    try:
        from ..services.table_schema_manager import get_all_cached_schemas
        cached_schemas = get_all_cached_schemas(db)
        
        if cached_schemas:
            table_info_parts = ["\n\nAvailable data tables for SQL queries:"]
            for schema in cached_schemas[:15]:  # Limit to top 15 tables
                table_name = schema["table_name"]
                row_count = schema.get("row_count", 0)
                date_range = schema.get("date_range")
                description = schema.get("description", "")
                
                info = f"- {table_name}"
                if description:
                    info += f" ({description[:80]})"
                info += f": {row_count:,} rows"
                
                if date_range:
                    info += f", date range {date_range.get('min_date', 'N/A')} to {date_range.get('max_date', 'N/A')}"
                
                table_info_parts.append(info)
            
            parts.append("\n".join(table_info_parts))
            logger.info(f"📊 Added {len(cached_schemas)} table availability info to context")
    except Exception as schema_err:
        logger.debug(f"Schema cache not available (not critical): {schema_err}")
    
    return "\n".join(parts) if parts else ""


@router.post("/ai-analysis/chat")
async def post_ai_analysis_chat(
    message: str = Body(..., embed=True),
    conversation_history: list = Body(default=[], embed=True),
    context_keys: list = Body(default=[], embed=True),
    days: int = Body(default=30, embed=True),
    time_scope: str = Body(default="current", embed=True),
    current_user: ZodiacUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generative AI analysis chat: answer user questions, optionally grounded in dashboard context.
    context_keys: stats, failed_summary, top_customers, inbound_summary, business_summary, process_flow.
    days: period for context (default 30).
    time_scope: 'current' (recent data), 'historical' (1994-2010), or 'both' (compare periods).
    Uses same config env vars as invoice-bot (OPENAI_API_KEY)."""
    ai_openai_key = _get_ai_analysis_config()
    if not ai_openai_key or not openai_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI analysis not available (set OPENAI_API_KEY, or Zodiac OPEN_AI_KEY)",
        )
    days = max(1, min(365, days)) if isinstance(days, (int, float)) else 30
    try:
        # 1) Build the existing dashboard context (Zodiac mode or SAP mode)
        if USE_SAP_DB_FOR_AI:
            from ..services.sap_ai_context import build_ai_context_from_sap
            sap_session_for_context = get_sap_session()
            context_str = ""
            if sap_session_for_context is not None:
                try:
                    context_str = build_ai_context_from_sap(
                        context_keys if isinstance(context_keys, list) else [],
                        sap_session_for_context,
                        days=int(days),
                    )
                except Exception as sap_e:
                    logger.warning("SAP AI context failed: %s", sap_e)
                finally:
                    sap_session_for_context.close()
        else:
            context_str = _build_ai_analysis_context(
                context_keys if isinstance(context_keys, list) else [],
                current_user,
                db,
                days=int(days),
            )

        # 2) INVOICE_BOT-like orchestrator: decide action, run SQL if needed, persist memory, and answer
        from ..services.ai_analysis_orchestrator import run_ai_analysis_orchestrator, orchestrator_payload

        sap_session_for_sql = get_sap_session() if USE_SAP_DB_FOR_AI else None
        try:
            orch = run_ai_analysis_orchestrator(
                api_key=ai_openai_key,
                user_id=current_user.id,
                user_query=message or "",
                db=db,
                conversation_history=conversation_history or [],
                context_str=context_str or "",
                sap_db=sap_session_for_sql,
                time_scope=time_scope or "current",
                days=int(days),
            )
        finally:
            if sap_session_for_sql is not None:
                sap_session_for_sql.close()

        return orchestrator_payload(orch)
    except Exception as e:
        logger.warning(f"AI analysis chat failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/ai-analysis-multi-model")
async def ai_analysis_multi_model_chat(
    message: str = Query(..., description="User's natural language query"),
    context_keys: Optional[list] = Query(None, description="Dashboard context keys to include"),
    days: int = Query(30, ge=1, le=365, description="Time period for context data"),
    time_scope: str = Query("current", description="Data scope: 'current', 'historical' (1994-2010), or 'both'"),
    current_user: ZodiacUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    AI analysis chat endpoint with multi-model comparison (GPT + Gemini + Claude).
    Runs all models in parallel and returns individual + synthesized responses.
    time_scope: 'current' (recent data), 'historical' (1994-2010), or 'both' (compare periods).
    """
    try:
        from ..config.config import ENABLE_MULTI_MODEL, USE_SAP_DB_FOR_AI
        from ..services.multi_model_orchestrator import run_all_models_parallel
        
        if not ENABLE_MULTI_MODEL:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Multi-model mode is not enabled. Set ENABLE_MULTI_MODEL=true in .env"
            )
        
        # Build context (same as regular AI analysis)
        if USE_SAP_DB_FOR_AI:
            from ..database import get_sap_session
            from ..services.sap_ai_context import build_ai_context_from_sap
            sap_session_for_context = get_sap_session()
            try:
                context_str = build_ai_context_from_sap(
                    context_keys if isinstance(context_keys, list) else [],
                    sap_session_for_context,
                    days=int(days),
                )
            except Exception as sap_e:
                logger.warning("SAP AI context failed: %s", sap_e)
                context_str = ""
            finally:
                sap_session_for_context.close()
        else:
            context_str = _build_ai_analysis_context(
                context_keys if isinstance(context_keys, list) else [],
                current_user,
                db,
                days=int(days),
            )
        
        # Run multi-model analysis
        result = await run_all_models_parallel(
            user_query=message,
            context=context_str,
            time_scope=time_scope,
            days=int(days),
        )
        
        return {
            "synthesized_answer": result.synthesized_answer,
            "best_model": result.best_model,
            "total_time_ms": result.total_time_ms,
            "time_scope": result.time_scope,
            "date_range": result.date_range,
            "period_info": result.period_info,
            "models": [
                {
                    "name": r.model_name,
                    "content": r.content,
                    "response_time_ms": r.response_time_ms,
                    "success": r.success,
                    "error": r.error,
                    "token_usage": r.token_usage,
                }
                for r in result.individual_responses
            ],
        }
    
    except Exception as e:
        logger.error(f"Multi-model analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/training-feedback")
async def submit_training_feedback(
    record_id: int = Query(..., description="Training data record ID"),
    feedback_score: int = Query(..., ge=1, le=5, description="Rating 1-5"),
    feedback_comment: str = Query(None, description="Optional feedback comment"),
    current_user: ZodiacUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Submit user feedback for an AI analysis response.
    Used to build training dataset for fine-tuning.
    """
    try:
        from ..services.training_data_collector import submit_feedback
        
        success = submit_feedback(
            db=db,
            record_id=record_id,
            feedback_score=feedback_score,
            feedback_comment=feedback_comment,
        )
        
        if success:
            return {"success": True, "message": "Feedback submitted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to submit feedback"
            )
    
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/voice-transcribe")
async def transcribe_voice(
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Query("en", description="Language code (e.g., 'en', 'es')"),
    current_user: ZodiacUser = Depends(get_current_user),
):
    """
    Transcribe audio file using OpenAI Whisper API.
    Supports WebM, MP3, WAV, and other common audio formats.
    """
    try:
        from ..services.voice_transcription import transcribe_audio
        
        # Read audio file content
        audio_content = await audio_file.read()
        
        if not audio_content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty audio file"
            )
        
        # Determine file format
        file_format = "webm"  # default
        if audio_file.filename:
            file_ext = audio_file.filename.split('.')[-1].lower()
            if file_ext in ['mp3', 'wav', 'm4a', 'flac', 'ogg', 'webm']:
                file_format = file_ext
        
        # Transcribe
        result = transcribe_audio(
            audio_file_content=audio_content,
            file_format=file_format,
            language=language if language != "auto" else None,
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return result
    
    except Exception as e:
        logger.error(f"Voice transcription failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/training-stats")
async def get_training_statistics(
    current_user: ZodiacUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get training data collection statistics for the current user.
    """
    try:
        from ..services.training_data_collector import get_training_stats
        
        stats = get_training_stats(db, user_id=current_user.id)
        return stats
    
    except Exception as e:
        logger.error(f"Failed to get training stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/auto-fix-details")
async def get_auto_fix_details(
    fix_type: str = Query(..., description="Type of fix to get details for"),
    current_user: ZodiacUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific auto-fix type:
    - Which invoices had this fix applied
    - Before and after values
    - Customer information
    - Success/failure status
    """
    try:
        logger.info(f"🔍 Fetching auto-fix details for: {fix_type}")
        
        # Map user-friendly names back to error types
        error_type_reverse_mapping = {
            'Missing Fields': ['missing_invoice_id', 'missing_field'],
            'Date Format': ['date_format', 'invalid_date'],
            'ID Padding': ['id_padding', 'invalid_id_format'],
            'Party Info': ['party_info', 'missing_party'],
        }
        
        error_types = error_type_reverse_mapping.get(fix_type, [fix_type.lower().replace(' ', '_')])
        
        # Query correction cache
        corrections = db.query(CorrectionCache).filter(
            CorrectionCache.error_type.in_(error_types),
            CorrectionCache.success_count > 0
        ).all()
        
        if not corrections:
            return {
                "fix_type": fix_type,
                "total_count": 0,
                "successful_count": 0,
                "failed_count": 0,
                "details": []
            }
        
        # Build detailed response
        details = []
        total_successful = 0
        total_failed = 0
        
        for correction in corrections:
            # Get transformation rule (contains before/after values)
            try:
                transform_rule = json.loads(correction.transformation_rule) if isinstance(
                    correction.transformation_rule, str
                ) else correction.transformation_rule
            except:
                transform_rule = {}
            
            # Get customer information
            customer_name = correction.customer_name or "Unknown Customer"
            
            # Extract before/after values
            before_value = transform_rule.get('original_value', 'null')
            after_value = transform_rule.get('corrected_value', 'N/A')
            fix_description = transform_rule.get('description', f"Applied {fix_type} fix")
            
            # Find a sample invoice that used this correction
            # Since invoices don't have customer_id, we find by user and check processing steps
            sample_invoice = db.query(SuccessModel).filter(
                SuccessModel.user_id == current_user.id,
                SuccessModel.processing_steps.isnot(None)
            ).order_by(SuccessModel.uploaded_at.desc()).first()
            
            if sample_invoice:
                # Parse processing steps to find AI corrections
                try:
                    steps = json.loads(sample_invoice.processing_steps) if isinstance(
                        sample_invoice.processing_steps, str
                    ) else sample_invoice.processing_steps
                    
                    # Look for AI correction step
                    for step in steps:
                        if step.get('step_name') == 'AI Correction' or 'AI' in step.get('message', ''):
                            error_details = step.get('error_details', {})
                            if isinstance(error_details, dict):
                                before_value = error_details.get('before', before_value)
                                after_value = error_details.get('after', after_value)
                                break
                except:
                    pass
            
            # Count successes and failures
            success_count = correction.success_count or 0
            failure_count = correction.failure_count or 0
            
            total_successful += success_count
            total_failed += failure_count
            
            # Create detail entry for each successful fix
            for i in range(min(success_count, 5)):  # Limit to 5 examples per correction type
                details.append({
                    "invoice_id": sample_invoice.id if sample_invoice else 0,
                    "tracking_id": sample_invoice.tracking_id if sample_invoice else f"N/A-{i}",
                    "customer_name": customer_name,
                    "fix_applied": fix_description,
                    "before_value": str(before_value),
                    "after_value": str(after_value),
                    "success": True,
                    "timestamp": (sample_invoice.uploaded_at if sample_invoice else datetime.utcnow()).isoformat()
                })
        
        # Sort by timestamp (most recent first)
        details.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Limit to top 20 details
        details = details[:20]
        
        return {
            "fix_type": fix_type,
            "total_count": total_successful + total_failed,
            "successful_count": total_successful,
            "failed_count": total_failed,
            "details": details
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch auto-fix details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch auto-fix details: {str(e)}"
        )


@router.get("/business")
async def get_business_analytics(
    days: int = Query(default=30, ge=1, le=365),
    current_user: ZodiacUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get business analytics for the dashboard Business tab:
    - E2E lifecycle funnel
    - Customer analysis (top customers, success rates, countries)
    - Country distribution
    - Product/Industry breakdown
    - Supplier analysis
    """
    try:
        logger.info(f"📊 Fetching business analytics for last {days} days")
        
        # Calculate date range
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Check if BI table has any data
        total_bi_records = db.query(func.count(InvoiceBusinessData.id)).filter(
            InvoiceBusinessData.user_id == current_user.id
        ).scalar() or 0
        
        if total_bi_records == 0:
            logger.warning(f"⚠️ No business intelligence data found for user {current_user.id}")
            
            # Check if user has any invoices at all
            total_invoices = (
                db.query(func.count(SuccessModel.id))
                .filter(SuccessModel.user_id == current_user.id)
                .scalar() or 0
            ) + (
                db.query(func.count(FailedModel.id))
                .filter(FailedModel.user_id == current_user.id)
                .scalar() or 0
            )
            
            if total_invoices > 0:
                logger.info(f"🔄 Auto-triggering backfill for {total_invoices} invoices")
                # Automatically trigger backfill in background
                try:
                    from ..services.business_intelligence_service import BusinessIntelligenceExtractor
                    from ..models.invoice_business_data import InvoiceBusinessData
                    import uuid
                    
                    bi_extractor = BusinessIntelligenceExtractor()
                    
                    # Process a limited batch to avoid timeout (first 25 invoices)
                    success_invoices = db.query(SuccessModel).filter(
                        SuccessModel.user_id == current_user.id
                    ).limit(25).all()
                    
                    failed_invoices = db.query(FailedModel).filter(
                        FailedModel.user_id == current_user.id
                    ).limit(25).all()
                    
                    processed = 0
                    errors = []
                    
                    # Process successful invoices
                    for invoice in success_invoices:
                        try:
                            xml_content = None
                            if invoice.edi_file_path:
                                try:
                                    xml_content = await read_file_from_storage(invoice.edi_file_path)
                                except Exception as read_err:
                                    logger.warning(f"Could not read file for invoice {invoice.id}: {read_err}")
                            
                            # Extract BI data
                            if xml_content:
                                bi_data = bi_extractor.extract_from_xml(xml_content)
                            else:
                                bi_data = {}
                            
                            # Determine lifecycle stage
                            current_stage = "SENT"
                            stage_status = "SUCCESS"
                            
                            # Create BI record
                            bi_record = InvoiceBusinessData(
                                tracking_id=invoice.tracking_id if hasattr(invoice, 'tracking_id') else uuid.uuid4(),
                                user_id=current_user.id,
                                success_invoice_id=invoice.id,
                                failed_invoice_id=None,
                                customer_id=bi_data.get('customer', {}).get('id'),
                                customer_name=bi_data.get('customer', {}).get('name'),
                                customer_country=bi_data.get('customer', {}).get('country'),
                                supplier_id=bi_data.get('supplier', {}).get('id'),
                                supplier_name=bi_data.get('supplier', {}).get('name'),
                                products=bi_data.get('products', []),
                                total_products_count=len(bi_data.get('products', [])),
                                total_amount=bi_data.get('financial', {}).get('total_amount'),
                                tax_amount=bi_data.get('financial', {}).get('tax_amount'),
                                currency=bi_data.get('financial', {}).get('currency'),
                                invoice_date=bi_data.get('financial', {}).get('invoice_date'),
                                industry=bi_data.get('industry', {}).get('name'),
                                industry_confidence=bi_data.get('industry', {}).get('confidence'),
                                current_stage=current_stage,
                                stage_status=stage_status,
                                source_file_format=invoice.file_format if hasattr(invoice, 'file_format') else None,
                                target_file_format=invoice.target_file_format if hasattr(invoice, 'target_file_format') else None
                            )
                            
                            db.add(bi_record)
                            db.commit()
                            processed += 1
                            
                        except Exception as e:
                            db.rollback()
                            error_msg = f"Invoice {invoice.id}: {str(e)}"
                            logger.error(f"Error processing invoice {invoice.id}: {e}")
                            errors.append(error_msg)
                            continue
                    
                    # Process failed invoices
                    for invoice in failed_invoices:
                        try:
                            xml_content = None
                            if invoice.edi_file_path:
                                try:
                                    xml_content = await read_file_from_storage(invoice.edi_file_path)
                                except Exception as read_err:
                                    logger.warning(f"Could not read file for invoice {invoice.id}: {read_err}")
                            
                            # Extract BI data
                            if xml_content:
                                bi_data = bi_extractor.extract_from_xml(xml_content)
                            else:
                                bi_data = {}
                            
                            # Determine failure stage
                            current_stage = "VALIDATED"
                            stage_status = "FAILED"
                            failed_at_stage = "VALIDATED"
                            
                            # Create BI record
                            bi_record = InvoiceBusinessData(
                                tracking_id=invoice.tracking_id if hasattr(invoice, 'tracking_id') else uuid.uuid4(),
                                user_id=current_user.id,
                                success_invoice_id=None,
                                failed_invoice_id=invoice.id,
                                customer_id=bi_data.get('customer', {}).get('id'),
                                customer_name=bi_data.get('customer', {}).get('name'),
                                customer_country=bi_data.get('customer', {}).get('country'),
                                supplier_id=bi_data.get('supplier', {}).get('id'),
                                supplier_name=bi_data.get('supplier', {}).get('name'),
                                products=bi_data.get('products', []),
                                total_products_count=len(bi_data.get('products', [])),
                                total_amount=bi_data.get('financial', {}).get('total_amount'),
                                tax_amount=bi_data.get('financial', {}).get('tax_amount'),
                                currency=bi_data.get('financial', {}).get('currency'),
                                invoice_date=bi_data.get('financial', {}).get('invoice_date'),
                                industry=bi_data.get('industry', {}).get('name'),
                                industry_confidence=bi_data.get('industry', {}).get('confidence'),
                                current_stage=current_stage,
                                stage_status=stage_status,
                                failed_at_stage=failed_at_stage,
                                failure_reason=invoice.error_message if hasattr(invoice, 'error_message') else None,
                                source_file_format=invoice.file_format if hasattr(invoice, 'file_format') else None,
                                target_file_format=invoice.target_file_format if hasattr(invoice, 'target_file_format') else None
                            )
                            
                            db.add(bi_record)
                            db.commit()
                            processed += 1
                            
                        except Exception as e:
                            db.rollback()
                            error_msg = f"Invoice {invoice.id}: {str(e)}"
                            logger.error(f"Error processing invoice {invoice.id}: {e}")
                            errors.append(error_msg)
                            continue
                    
                    logger.info(f"✅ Auto-backfill processed {processed} invoices (errors: {len(errors)})")
                    
                    if processed > 0:
                        return {
                            "message": f"Successfully processed {processed} of {total_invoices} invoices! Refresh to see your analytics.",
                            "needs_backfill": False,  # Set to false so it shows data on next refresh
                            "auto_backfill_triggered": True,
                            "processed_count": processed,
                            "total_invoices": total_invoices,
                            "errors_count": len(errors),
                            "lifecycle_funnel": {
                                'RECEIVED': {'total': 0, 'success': 0, 'failed': 0},
                                'VALIDATED': {'total': 0, 'success': 0, 'failed': 0},
                                'CONVERTED': {'total': 0, 'success': 0, 'failed': 0},
                                'SENT': {'total': 0, 'success': 0, 'failed': 0},
                                'ACKNOWLEDGED': {'total': 0, 'success': 0, 'failed': 0},
                            },
                            "customer_analysis": {"top_customers": [], "total_customers": 0},
                            "country_distribution": [],
                            "industry_breakdown": [],
                            "product_analysis": {"top_products": [], "total_products": 0},
                            "supplier_analysis": {"top_suppliers": []}
                        }
                    else:
                        logger.error(f"❌ No invoices processed. Errors: {errors[:5]}")
                        
                except Exception as e:
                    logger.error(f"❌ Auto-backfill failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # Return empty structure with helpful message
            return {
                "message": "No business data available yet. Upload invoices or SAT documents to see business analytics." if total_invoices == 0 else "Processing your invoices... Please refresh in a moment.",
                "needs_backfill": total_invoices > 0,
                "has_data": False,
                "lifecycle_funnel": {
                    'RECEIVED': {'total': 0, 'success': 0, 'failed': 0},
                    'VALIDATED': {'total': 0, 'success': 0, 'failed': 0},
                    'CONVERTED': {'total': 0, 'success': 0, 'failed': 0},
                    'SENT': {'total': 0, 'success': 0, 'failed': 0},
                    'ACKNOWLEDGED': {'total': 0, 'success': 0, 'failed': 0},
                },
                "customer_analysis": {"top_customers": [], "total_customers": 0},
                "country_distribution": [],
                "industry_breakdown": [],
                "product_analysis": {"top_products": [], "total_products": 0},
                "supplier_analysis": {"top_suppliers": []}
            }
        
        logger.info(f"✅ Found {total_bi_records} legacy business intelligence records")
        
        # Check for Invoice V2 BI records
        total_v2_bi_records = db.query(func.count(InvoiceV2BusinessData.id)).filter(
            InvoiceV2BusinessData.user_id == current_user.id
        ).scalar() or 0
        
        logger.info(f"✅ Found {total_v2_bi_records} Invoice V2 business intelligence records")
        
        # ============================================================
        # 1. E2E LIFECYCLE FUNNEL
        # ============================================================
        # Count invoices at each stage (legacy)
        stage_counts = db.query(
            InvoiceBusinessData.current_stage,
            InvoiceBusinessData.stage_status,
            func.count(InvoiceBusinessData.id).label('count')
        ).filter(
            InvoiceBusinessData.user_id == current_user.id,
            InvoiceBusinessData.created_at >= cutoff_date
        ).group_by(
            InvoiceBusinessData.current_stage,
            InvoiceBusinessData.stage_status
        ).all()
        
        # Count Invoice V2 invoices at each stage
        v2_stage_counts = db.query(
            InvoiceV2BusinessData.current_stage,
            InvoiceV2BusinessData.stage_status,
            func.count(InvoiceV2BusinessData.id).label('count')
        ).filter(
            InvoiceV2BusinessData.user_id == current_user.id,
            InvoiceV2BusinessData.created_at >= cutoff_date
        ).group_by(
            InvoiceV2BusinessData.current_stage,
            InvoiceV2BusinessData.stage_status
        ).all()
        
        # Build lifecycle funnel (combine legacy + V2)
        lifecycle_funnel = {
            'RECEIVED': {'total': 0, 'success': 0, 'failed': 0},
            'VALIDATED': {'total': 0, 'success': 0, 'failed': 0},
            'CONVERTED': {'total': 0, 'success': 0, 'failed': 0},
            'SENT': {'total': 0, 'success': 0, 'failed': 0},
            'ACKNOWLEDGED': {'total': 0, 'success': 0, 'failed': 0},
        }
        
        # Add legacy data
        for stage, status, count in stage_counts:
            if stage in lifecycle_funnel:
                lifecycle_funnel[stage]['total'] += count
                if status == 'SUCCESS':
                    lifecycle_funnel[stage]['success'] += count
                elif status == 'FAILED':
                    lifecycle_funnel[stage]['failed'] += count
        
        # Add V2 data
        for stage, status, count in v2_stage_counts:
            if stage in lifecycle_funnel:
                lifecycle_funnel[stage]['total'] += count
                if status == 'SUCCESS':
                    lifecycle_funnel[stage]['success'] += count
                elif status == 'FAILED':
                    lifecycle_funnel[stage]['failed'] += count
        
        # ============================================================
        # 2. CUSTOMER ANALYSIS
        # ============================================================
        # Top customers with success/failed counts (legacy)
        customer_stats = db.query(
            InvoiceBusinessData.customer_id,
            InvoiceBusinessData.customer_name,
            InvoiceBusinessData.customer_country,
            InvoiceBusinessData.stage_status,
            func.count(InvoiceBusinessData.id).label('count'),
            func.sum(InvoiceBusinessData.total_amount).label('total_revenue')
        ).filter(
            InvoiceBusinessData.user_id == current_user.id,
            InvoiceBusinessData.created_at >= cutoff_date,
            InvoiceBusinessData.customer_name.isnot(None)
        ).group_by(
            InvoiceBusinessData.customer_id,
            InvoiceBusinessData.customer_name,
            InvoiceBusinessData.customer_country,
            InvoiceBusinessData.stage_status
        ).all()
        
        # Top customers (Invoice V2)
        v2_customer_stats = db.query(
            InvoiceV2BusinessData.customer_id,
            InvoiceV2BusinessData.customer_name,
            InvoiceV2BusinessData.customer_country,
            InvoiceV2BusinessData.stage_status,
            func.count(InvoiceV2BusinessData.id).label('count'),
            func.sum(InvoiceV2BusinessData.total_amount).label('total_revenue')
        ).filter(
            InvoiceV2BusinessData.user_id == current_user.id,
            InvoiceV2BusinessData.created_at >= cutoff_date,
            InvoiceV2BusinessData.customer_name.isnot(None)
        ).group_by(
            InvoiceV2BusinessData.customer_id,
            InvoiceV2BusinessData.customer_name,
            InvoiceV2BusinessData.customer_country,
            InvoiceV2BusinessData.stage_status
        ).all()
        
        # Aggregate customer data (legacy + V2)
        customer_map = {}
        
        # Add legacy data
        for cust_id, cust_name, country, status, count, revenue in customer_stats:
            key = cust_id or cust_name
            if key not in customer_map:
                customer_map[key] = {
                    'customer_id': cust_id,
                    'customer_name': cust_name,
                    'customer_country': country,
                    'total_invoices': 0,
                    'successful': 0,
                    'failed': 0,
                    'total_revenue': 0
                }
            
            customer_map[key]['total_invoices'] += count
            if status == 'SUCCESS':
                customer_map[key]['successful'] += count
            elif status == 'FAILED':
                customer_map[key]['failed'] += count
            customer_map[key]['total_revenue'] += float(revenue or 0)
        
        # Add V2 data
        for cust_id, cust_name, country, status, count, revenue in v2_customer_stats:
            key = cust_id or cust_name
            if key not in customer_map:
                customer_map[key] = {
                    'customer_id': cust_id,
                    'customer_name': cust_name,
                    'customer_country': country,
                    'total_invoices': 0,
                    'successful': 0,
                    'failed': 0,
                    'total_revenue': 0
                }
            
            customer_map[key]['total_invoices'] += count
            if status == 'SUCCESS':
                customer_map[key]['successful'] += count
            elif status == 'FAILED':
                customer_map[key]['failed'] += count
            customer_map[key]['total_revenue'] += float(revenue or 0)
        
        # Convert to list and calculate success rates
        customer_list = []
        for customer in customer_map.values():
            success_rate = 0
            if customer['total_invoices'] > 0:
                success_rate = round((customer['successful'] / customer['total_invoices']) * 100, 1)
            
            customer_list.append({
                **customer,
                'success_rate': success_rate
            })
        
        # Sort by total invoices and get top 10
        top_customers = sorted(customer_list, key=lambda x: x['total_invoices'], reverse=True)[:10]
        
        # ============================================================
        # 3. COUNTRY DISTRIBUTION
        # ============================================================
        # Legacy data
        country_stats = db.query(
            InvoiceBusinessData.customer_country,
            func.count(InvoiceBusinessData.id).label('count')
        ).filter(
            InvoiceBusinessData.user_id == current_user.id,
            InvoiceBusinessData.created_at >= cutoff_date,
            InvoiceBusinessData.customer_country.isnot(None)
        ).group_by(
            InvoiceBusinessData.customer_country
        ).all()
        
        # V2 data
        v2_country_stats = db.query(
            InvoiceV2BusinessData.customer_country,
            func.count(InvoiceV2BusinessData.id).label('count')
        ).filter(
            InvoiceV2BusinessData.user_id == current_user.id,
            InvoiceV2BusinessData.created_at >= cutoff_date,
            InvoiceV2BusinessData.customer_country.isnot(None)
        ).group_by(
            InvoiceV2BusinessData.customer_country
        ).all()
        
        # Combine and aggregate
        country_map = {}
        for country, count in country_stats:
            country_map[country] = country_map.get(country, 0) + count
        for country, count in v2_country_stats:
            country_map[country] = country_map.get(country, 0) + count
        
        # Sort and limit
        country_distribution = [
            {'country': country, 'count': count}
            for country, count in sorted(country_map.items(), key=lambda x: x[1], reverse=True)[:15]
        ]
        
        # ============================================================
        # 4. INDUSTRY BREAKDOWN
        # ============================================================
        # Legacy data
        industry_stats = db.query(
            InvoiceBusinessData.industry,
            func.count(InvoiceBusinessData.id).label('count'),
            func.sum(InvoiceBusinessData.total_amount).label('total_revenue')
        ).filter(
            InvoiceBusinessData.user_id == current_user.id,
            InvoiceBusinessData.created_at >= cutoff_date,
            InvoiceBusinessData.industry.isnot(None)
        ).group_by(
            InvoiceBusinessData.industry
        ).all()
        
        # V2 data
        v2_industry_stats = db.query(
            InvoiceV2BusinessData.industry,
            func.count(InvoiceV2BusinessData.id).label('count'),
            func.sum(InvoiceV2BusinessData.total_amount).label('total_revenue')
        ).filter(
            InvoiceV2BusinessData.user_id == current_user.id,
            InvoiceV2BusinessData.created_at >= cutoff_date,
            InvoiceV2BusinessData.industry.isnot(None)
        ).group_by(
            InvoiceV2BusinessData.industry
        ).all()
        
        # Combine and aggregate
        industry_map = {}
        for industry, count, revenue in industry_stats:
            if industry not in industry_map:
                industry_map[industry] = {'count': 0, 'total_revenue': 0}
            industry_map[industry]['count'] += count
            industry_map[industry]['total_revenue'] += float(revenue or 0)
        
        for industry, count, revenue in v2_industry_stats:
            if industry not in industry_map:
                industry_map[industry] = {'count': 0, 'total_revenue': 0}
            industry_map[industry]['count'] += count
            industry_map[industry]['total_revenue'] += float(revenue or 0)
        
        # Sort by count
        industry_breakdown = [
            {
                'industry': industry,
                'count': data['count'],
                'total_revenue': data['total_revenue']
            }
            for industry, data in sorted(industry_map.items(), key=lambda x: x[1]['count'], reverse=True)
        ]
        
        # ============================================================
        # 5. PRODUCT ANALYSIS (Top products across all invoices)
        # ============================================================
        # This requires parsing the products JSONB field
        product_counts = {}
        
        # Legacy products
        bi_records_with_products = db.query(InvoiceBusinessData).filter(
            InvoiceBusinessData.user_id == current_user.id,
            InvoiceBusinessData.created_at >= cutoff_date,
            InvoiceBusinessData.products.isnot(None)
        ).all()
        
        for record in bi_records_with_products:
            products = record.products
            if products and isinstance(products, list):
                for product in products:
                    product_name = product.get('name', 'Unknown')
                    if product_name not in product_counts:
                        product_counts[product_name] = {
                            'name': product_name,
                            'count': 0,
                            'total_quantity': 0,
                            'total_revenue': 0
                        }
                    product_counts[product_name]['count'] += 1
                    product_counts[product_name]['total_quantity'] += float(product.get('quantity', 0))
                    product_counts[product_name]['total_revenue'] += float(product.get('line_total', 0))
        
        # V2 products
        v2_bi_records_with_products = db.query(InvoiceV2BusinessData).filter(
            InvoiceV2BusinessData.user_id == current_user.id,
            InvoiceV2BusinessData.created_at >= cutoff_date,
            InvoiceV2BusinessData.products.isnot(None)
        ).all()
        
        for record in v2_bi_records_with_products:
            products = record.products
            if products and isinstance(products, list):
                for product in products:
                    product_name = product.get('name', 'Unknown')
                    if product_name not in product_counts:
                        product_counts[product_name] = {
                            'name': product_name,
                            'count': 0,
                            'total_quantity': 0,
                            'total_revenue': 0
                        }
                    product_counts[product_name]['count'] += 1
                    product_counts[product_name]['total_quantity'] += float(product.get('quantity', 0))
                    product_counts[product_name]['total_revenue'] += float(product.get('revenue', 0))
        
        # Sort by count and get top 10
        top_products = sorted(product_counts.values(), key=lambda x: x['count'], reverse=True)[:10]
        
        # ============================================================
        # 6. SUPPLIER ANALYSIS
        # ============================================================
        # Legacy suppliers
        supplier_stats = db.query(
            InvoiceBusinessData.supplier_id,
            InvoiceBusinessData.supplier_name,
            func.count(InvoiceBusinessData.id).label('count')
        ).filter(
            InvoiceBusinessData.user_id == current_user.id,
            InvoiceBusinessData.created_at >= cutoff_date,
            InvoiceBusinessData.supplier_name.isnot(None)
        ).group_by(
            InvoiceBusinessData.supplier_id,
            InvoiceBusinessData.supplier_name
        ).all()
        
        # V2 suppliers
        v2_supplier_stats = db.query(
            InvoiceV2BusinessData.supplier_id,
            InvoiceV2BusinessData.supplier_name,
            func.count(InvoiceV2BusinessData.id).label('count')
        ).filter(
            InvoiceV2BusinessData.user_id == current_user.id,
            InvoiceV2BusinessData.created_at >= cutoff_date,
            InvoiceV2BusinessData.supplier_name.isnot(None)
        ).group_by(
            InvoiceV2BusinessData.supplier_id,
            InvoiceV2BusinessData.supplier_name
        ).all()
        
        # Combine and aggregate
        supplier_map = {}
        for supplier_id, supplier_name, count in supplier_stats:
            key = supplier_id or supplier_name
            if key not in supplier_map:
                supplier_map[key] = {
                    'supplier_id': supplier_id,
                    'supplier_name': supplier_name,
                    'count': 0
                }
            supplier_map[key]['count'] += count
        
        for supplier_id, supplier_name, count in v2_supplier_stats:
            key = supplier_id or supplier_name
            if key not in supplier_map:
                supplier_map[key] = {
                    'supplier_id': supplier_id,
                    'supplier_name': supplier_name,
                    'count': 0
                }
            supplier_map[key]['count'] += count
        
        # Sort and limit to top 10
        top_suppliers = sorted(supplier_map.values(), key=lambda x: x['count'], reverse=True)[:10]
        
        # ============================================================
        # RETURN RESPONSE
        # ============================================================
        return {
            "lifecycle_funnel": lifecycle_funnel,
            "customer_analysis": {
                "top_customers": top_customers,
                "total_customers": len(customer_list)
            },
            "country_distribution": country_distribution,
            "industry_breakdown": industry_breakdown,
            "product_analysis": {
                "top_products": top_products,
                "total_products": len(product_counts)
            },
            "supplier_analysis": {
                "top_suppliers": top_suppliers
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch business analytics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return empty data instead of 500 error
        return {
            "lifecycle_funnel": {
                'RECEIVED': {'total': 0, 'success': 0, 'failed': 0},
                'VALIDATED': {'total': 0, 'success': 0, 'failed': 0},
                'CONVERTED': {'total': 0, 'success': 0, 'failed': 0},
                'SENT': {'total': 0, 'success': 0, 'failed': 0},
                'ACKNOWLEDGED': {'total': 0, 'success': 0, 'failed': 0},
            },
            "customer_analysis": {"top_customers": [], "total_customers": 0, "by_country": []},
            "country_distribution": [],
            "industry_breakdown": [],
            "product_analysis": {"top_products": [], "total_products": 0},
            "supplier_analysis": {"top_suppliers": []}
        }


@router.get("/industry-intelligence")
async def get_industry_intelligence(
    days: int = Query(default=30, ge=1, le=365),
    current_user: ZodiacUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get Product Performance & Industry Intelligence:
    - Product performance metrics with industry benchmarks
    - Performance indicators (underperforming/optimal/outperforming)
    - AI-powered insights and recommendations
    - Industry standards and competitive positioning
    """
    try:
        logger.info(f"📊 Fetching industry intelligence for last {days} days")
        
        # Calculate date range
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get all BI records with products
        bi_records = db.query(InvoiceBusinessData).filter(
            InvoiceBusinessData.user_id == current_user.id,
            InvoiceBusinessData.created_at >= cutoff_date,
            InvoiceBusinessData.products.isnot(None)
        ).all()
        
        if not bi_records:
            logger.warning(f"⚠️ No product data found for user {current_user.id}")
            return {
                "message": "No product data available yet. Upload invoices with product information to see industry intelligence.",
                "has_data": False,
                "summary": {
                    "total_products": 0,
                    "underperforming": 0,
                    "optimal": 0,
                    "outperforming": 0,
                    "total_revenue": 0
                },
                "products": [],
                "industry_benchmarks": {},
                "ai_insights": None,
                "date_range": {
                    "start": cutoff_date.isoformat(),
                    "end": datetime.utcnow().isoformat(),
                    "days": days
                }
            }
        
        # ============================================================
        # 1. AGGREGATE PRODUCT DATA
        # ============================================================
        product_data = defaultdict(lambda: {
            'name': '',
            'industry': 'General',
            'total_quantity': 0,
            'total_revenue': 0,
            'order_count': 0,
            'prices': [],
            'invoices': []
        })
        
        industry_products = defaultdict(lambda: defaultdict(list))
        
        for record in bi_records:
            industry = record.industry or 'General'
            products = record.products if isinstance(record.products, list) else []
            
            for product in products:
                product_name = product.get('name', 'Unknown')
                quantity = float(product.get('quantity', 0))
                unit_price = float(product.get('unit_price', 0))
                line_total = float(product.get('line_total', 0))
                
                # Aggregate by product name
                product_data[product_name]['name'] = product_name
                product_data[product_name]['industry'] = industry
                product_data[product_name]['total_quantity'] += quantity
                product_data[product_name]['total_revenue'] += line_total
                product_data[product_name]['order_count'] += 1
                if unit_price > 0:
                    product_data[product_name]['prices'].append(unit_price)
                product_data[product_name]['invoices'].append(record.tracking_id)
                
                # Track for industry benchmarking
                if unit_price > 0:
                    industry_products[industry][product_name].append({
                        'price': unit_price,
                        'quantity': quantity,
                        'revenue': line_total
                    })
        
        # ============================================================
        # 2. CALCULATE INDUSTRY BENCHMARKS
        # ============================================================
        industry_benchmarks = {}
        
        for industry, products in industry_products.items():
            all_prices = []
            all_revenues = []
            all_quantities = []
            
            for product_name, product_list in products.items():
                for p in product_list:
                    all_prices.append(p['price'])
                    all_revenues.append(p['revenue'])
                    all_quantities.append(p['quantity'])
            
            if all_prices:
                industry_benchmarks[industry] = {
                    'avg_price': round(sum(all_prices) / len(all_prices), 2),
                    'median_price': round(sorted(all_prices)[len(all_prices) // 2], 2),
                    'avg_revenue': round(sum(all_revenues) / len(all_revenues), 2),
                    'avg_quantity': round(sum(all_quantities) / len(all_quantities), 2),
                    'total_products': len(products),
                    'price_std_dev': round(_calculate_std_dev(all_prices), 2)
                }
        
        # ============================================================
        # 3. ANALYZE PRODUCT PERFORMANCE
        # ============================================================
        products_analysis = []
        
        for product_name, data in product_data.items():
            industry = data['industry']
            avg_price = sum(data['prices']) / len(data['prices']) if data['prices'] else 0
            
            # Get industry benchmark
            benchmark = industry_benchmarks.get(industry, {})
            industry_avg_price = benchmark.get('avg_price', avg_price)
            industry_avg_revenue = benchmark.get('avg_revenue', data['total_revenue'] / data['order_count'] if data['order_count'] > 0 else 0)
            
            # Calculate performance metrics
            price_diff_pct = 0
            if industry_avg_price > 0:
                price_diff_pct = round(((avg_price - industry_avg_price) / industry_avg_price) * 100, 1)
            
            avg_revenue_per_order = data['total_revenue'] / data['order_count'] if data['order_count'] > 0 else 0
            revenue_diff_pct = 0
            if industry_avg_revenue > 0:
                revenue_diff_pct = round(((avg_revenue_per_order - industry_avg_revenue) / industry_avg_revenue) * 100, 1)
            
            # Determine performance status
            performance_status = _determine_performance_status(
                price_diff_pct, 
                revenue_diff_pct, 
                data['order_count'],
                benchmark.get('total_products', 1)
            )
            
            # Price positioning
            price_position = 'Medium'
            if avg_price > industry_avg_price * 1.2:
                price_position = 'Premium'
            elif avg_price < industry_avg_price * 0.8:
                price_position = 'Economy'
            
            products_analysis.append({
                'product_name': product_name,
                'industry': industry,
                'performance_status': performance_status,
                'performance_score': _calculate_performance_score(price_diff_pct, revenue_diff_pct, data['order_count']),
                'metrics': {
                    'avg_price': round(avg_price, 2),
                    'total_quantity': round(data['total_quantity'], 2),
                    'total_revenue': round(data['total_revenue'], 2),
                    'order_count': data['order_count'],
                    'avg_revenue_per_order': round(avg_revenue_per_order, 2),
                    'price_position': price_position
                },
                'benchmarks': {
                    'industry_avg_price': round(industry_avg_price, 2),
                    'industry_avg_revenue': round(industry_avg_revenue, 2),
                    'price_diff_pct': price_diff_pct,
                    'revenue_diff_pct': revenue_diff_pct
                },
                'trend': _analyze_trend(data['invoices'], bi_records)
            })
        
        # Sort by performance score (worst first for attention)
        products_analysis.sort(key=lambda x: x['performance_score'])
        
        # ============================================================
        # 4. GENERATE AI INSIGHTS (if available)
        # ============================================================
        ai_insights = []
        
        if OPENAI_API_KEY:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                
                # Analyze top underperforming and top performing products
                underperforming = [p for p in products_analysis if p['performance_status'] == 'underperforming'][:3]
                outperforming = [p for p in products_analysis if p['performance_status'] == 'outperforming'][:3]
                
                if underperforming or outperforming:
                    prompt = f"""Analyze these product performance metrics and provide actionable business insights:

UNDERPERFORMING PRODUCTS:
{json.dumps(underperforming, indent=2)}

OUTPERFORMING PRODUCTS:
{json.dumps(outperforming, indent=2)}

INDUSTRY BENCHMARKS:
{json.dumps(industry_benchmarks, indent=2)}

Provide insights in JSON format:
{{
  "overall_insights": ["insight 1", "insight 2", "insight 3"],
  "product_recommendations": [
    {{
      "product_name": "Product Name",
      "issue": "What's wrong",
      "recommendation": "Specific action to take",
      "expected_impact": "Expected result"
    }}
  ],
  "industry_trends": ["trend 1", "trend 2"]
}}

Focus on: pricing strategy, demand patterns, competitive positioning, and revenue optimization."""

                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        response_format={"type": "json_object"}
                    )
                    
                    ai_response = json.loads(completion.choices[0].message.content)
                    ai_insights = ai_response
                    
                    logger.info(f"✅ AI insights generated for {len(products_analysis)} products")
            
            except Exception as ai_error:
                logger.warning(f"⚠️ AI insights generation failed: {ai_error}")
        
        # ============================================================
        # 5. RETURN COMPREHENSIVE ANALYSIS
        # ============================================================
        return {
            "summary": {
                "total_products": len(products_analysis),
                "underperforming": len([p for p in products_analysis if p['performance_status'] == 'underperforming']),
                "optimal": len([p for p in products_analysis if p['performance_status'] == 'optimal']),
                "outperforming": len([p for p in products_analysis if p['performance_status'] == 'outperforming']),
                "total_revenue": round(sum(p['metrics']['total_revenue'] for p in products_analysis), 2)
            },
            "products": products_analysis,
            "industry_benchmarks": industry_benchmarks,
            "ai_insights": ai_insights,
            "date_range": {
                "start": cutoff_date.isoformat(),
                "end": datetime.utcnow().isoformat(),
                "days": days
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch industry intelligence: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch industry intelligence: {str(e)}"
        )


# Common ISO 3166-1 alpha-2 country codes to full names (for display)
COUNTRY_CODE_NAMES = {
    "NZ": "New Zealand", "AU": "Australia", "US": "United States", "GB": "United Kingdom", "UK": "United Kingdom",
    "DE": "Germany", "FR": "France", "JP": "Japan", "CN": "China", "IN": "India", "SG": "Singapore",
    "MY": "Malaysia", "TH": "Thailand", "ID": "Indonesia", "PH": "Philippines", "VN": "Vietnam",
    "KR": "South Korea", "CA": "Canada", "MX": "Mexico", "BR": "Brazil", "ES": "Spain", "IT": "Italy",
    "NL": "Netherlands", "CH": "Switzerland", "SE": "Sweden", "NO": "Norway", "DK": "Denmark",
    "FI": "Finland", "IE": "Ireland", "BE": "Belgium", "AT": "Austria", "PL": "Poland",
    "AE": "United Arab Emirates", "SA": "Saudi Arabia", "ZA": "South Africa", "HK": "Hong Kong",
}


def _country_code_to_name(code: str) -> str:
    """Return full country name for ISO code, or code itself if unknown."""
    if not code or code == "Unknown":
        return code or "Unknown"
    return COUNTRY_CODE_NAMES.get(str(code).upper(), code)


def _calculate_std_dev(values):
    """Calculate standard deviation"""
    if not values:
        return 0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def _determine_performance_status(price_diff_pct, revenue_diff_pct, order_count, total_products):
    """Determine if product is underperforming, optimal, or outperforming"""
    # Score based on multiple factors
    score = 0
    
    # Revenue performance (most important)
    if revenue_diff_pct > 20:
        score += 2
    elif revenue_diff_pct > 0:
        score += 1
    elif revenue_diff_pct < -20:
        score -= 2
    else:
        score -= 1
    
    # Order frequency
    avg_orders = total_products / 3 if total_products > 3 else 1
    if order_count > avg_orders * 1.5:
        score += 1
    elif order_count < avg_orders * 0.5:
        score -= 1
    
    # Determine status
    if score >= 2:
        return 'outperforming'
    elif score <= -2:
        return 'underperforming'
    else:
        return 'optimal'


def _calculate_performance_score(price_diff_pct, revenue_diff_pct, order_count):
    """Calculate overall performance score (lower is worse, for sorting)"""
    # Negative score for underperformance
    score = revenue_diff_pct + (order_count * 5)
    return score


def _analyze_trend(invoices, all_records):
    """Analyze if product demand is increasing, stable, or declining"""
    # Simple trend analysis based on recent vs older invoices
    if len(invoices) < 2:
        return 'insufficient_data'
    
    # Get timestamps
    timestamps = []
    for record in all_records:
        if record.tracking_id in invoices:
            timestamps.append(record.created_at)
    
    timestamps.sort()
    
    if len(timestamps) < 2:
        return 'stable'
    
    # Compare first half vs second half
    mid = len(timestamps) // 2
    first_half = timestamps[:mid]
    second_half = timestamps[mid:]
    
    if len(second_half) > len(first_half) * 1.2:
        return 'increasing'
    elif len(second_half) < len(first_half) * 0.8:
        return 'declining'
    else:
        return 'stable'


# ============================================================================
# Invoice V2 Business Intelligence Endpoints
# ============================================================================

@router.get("/revenue-analysis")
async def get_revenue_analysis(
    days: int = Query(default=90, ge=1, le=365),
    current_user: ZodiacUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get revenue analysis by country, season, and fiscal quarter.
    Combines legacy and Invoice V2 data.
    """
    logger.info(f"📊 Fetching revenue analysis for last {days} days")
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # ============================================================
        # REVENUE BY COUNTRY
        # ============================================================
        # Legacy data
        country_revenue_legacy = db.query(
            InvoiceBusinessData.customer_country,
            func.count(InvoiceBusinessData.id).label('invoice_count'),
            func.sum(InvoiceBusinessData.total_amount).label('revenue')
        ).filter(
            InvoiceBusinessData.user_id == current_user.id,
            InvoiceBusinessData.created_at >= cutoff_date,
            InvoiceBusinessData.customer_country.isnot(None)
        ).group_by(
            InvoiceBusinessData.customer_country
        ).all()
        
        # V2 data
        country_revenue_v2 = db.query(
            InvoiceV2BusinessData.customer_country,
            func.count(InvoiceV2BusinessData.id).label('invoice_count'),
            func.sum(InvoiceV2BusinessData.total_amount).label('revenue')
        ).filter(
            InvoiceV2BusinessData.user_id == current_user.id,
            InvoiceV2BusinessData.created_at >= cutoff_date,
            InvoiceV2BusinessData.customer_country.isnot(None)
        ).group_by(
            InvoiceV2BusinessData.customer_country
        ).all()
        
        # Combine
        country_map = {}
        for country, count, revenue in country_revenue_legacy:
            country_map[country] = {
                'country': country,
                'revenue': float(revenue or 0),
                'invoice_count': count
            }
        
        for country, count, revenue in country_revenue_v2:
            if country not in country_map:
                country_map[country] = {
                    'country': country,
                    'revenue': 0,
                    'invoice_count': 0
                }
            country_map[country]['revenue'] += float(revenue or 0)
            country_map[country]['invoice_count'] += count
        
        by_country = sorted(country_map.values(), key=lambda x: x['revenue'], reverse=True)
        
        # ============================================================
        # REVENUE BY SEASON
        # ============================================================
        # V2 data (has season field)
        season_revenue = db.query(
            InvoiceV2BusinessData.season,
            func.sum(InvoiceV2BusinessData.total_amount).label('revenue'),
            func.count(InvoiceV2BusinessData.id).label('invoice_count')
        ).filter(
            InvoiceV2BusinessData.user_id == current_user.id,
            InvoiceV2BusinessData.created_at >= cutoff_date,
            InvoiceV2BusinessData.season.isnot(None)
        ).group_by(
            InvoiceV2BusinessData.season
        ).all()
        
        by_season = [
            {
                'season': season,
                'revenue': float(revenue or 0),
                'invoice_count': count
            }
            for season, revenue, count in season_revenue
        ]
        
        # ============================================================
        # REVENUE BY FISCAL QUARTER
        # ============================================================
        # V2 data (has fiscal quarter field)
        quarter_revenue = db.query(
            InvoiceV2BusinessData.fiscal_quarter,
            InvoiceV2BusinessData.fiscal_year,
            func.sum(InvoiceV2BusinessData.total_amount).label('revenue'),
            func.count(InvoiceV2BusinessData.id).label('invoice_count')
        ).filter(
            InvoiceV2BusinessData.user_id == current_user.id,
            InvoiceV2BusinessData.created_at >= cutoff_date,
            InvoiceV2BusinessData.fiscal_quarter.isnot(None)
        ).group_by(
            InvoiceV2BusinessData.fiscal_quarter,
            InvoiceV2BusinessData.fiscal_year
        ).order_by(
            InvoiceV2BusinessData.fiscal_year.desc(),
            InvoiceV2BusinessData.fiscal_quarter.desc()
        ).all()
        
        by_quarter = [
            {
                'quarter': quarter,
                'year': year,
                'revenue': float(revenue or 0),
                'invoice_count': count
            }
            for quarter, year, revenue, count in quarter_revenue
        ]
        
        return {
            "by_country": by_country,
            "by_season": by_season,
            "by_quarter": by_quarter,
            "period_days": days
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch revenue analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Revenue analysis failed: {str(e)}"
        )


@router.get("/product-demand")
async def get_product_demand_analysis(
    days: int = Query(default=90, ge=1, le=365),
    current_user: ZodiacUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get product demand analysis including:
    - Trending products (increasing/stable/decreasing)
    - Top customers per product
    - Top countries per product
    """
    logger.info(f"📊 Fetching product demand analysis for last {days} days")
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get all V2 BI records with products
        v2_records = db.query(InvoiceV2BusinessData).filter(
            InvoiceV2BusinessData.user_id == current_user.id,
            InvoiceV2BusinessData.created_at >= cutoff_date,
            InvoiceV2BusinessData.products.isnot(None)
        ).order_by(
            InvoiceV2BusinessData.invoice_date.desc()
        ).all()
        
        # Aggregate product data
        product_data = {}
        customer_product_map = {}  # Track which customers buy which products
        country_product_map = {}   # Track which countries buy which products
        
        for record in v2_records:
            products = record.products
            if not products or not isinstance(products, list):
                continue
            
            for product in products:
                product_name = product.get('name', 'Unknown')
                
                # Initialize product data
                if product_name not in product_data:
                    product_data[product_name] = {
                        'name': product_name,
                        'total_quantity': 0,
                        'total_revenue': 0,
                        'order_count': 0,
                        'customers': set(),
                        'countries': set(),
                        'monthly_counts': {}
                    }
                
                # Aggregate metrics
                product_data[product_name]['total_quantity'] += float(product.get('quantity', 0))
                product_data[product_name]['total_revenue'] += float(product.get('revenue', 0))
                product_data[product_name]['order_count'] += 1
                
                if record.customer_name:
                    product_data[product_name]['customers'].add(record.customer_name)
                    
                    # Track customer-product relationship
                    if record.customer_name not in customer_product_map:
                        customer_product_map[record.customer_name] = {}
                    if product_name not in customer_product_map[record.customer_name]:
                        customer_product_map[record.customer_name][product_name] = 0
                    customer_product_map[record.customer_name][product_name] += 1
                
                if record.customer_country:
                    product_data[product_name]['countries'].add(record.customer_country)
                    
                    # Track country-product relationship
                    if record.customer_country not in country_product_map:
                        country_product_map[record.customer_country] = {}
                    if product_name not in country_product_map[record.customer_country]:
                        country_product_map[record.customer_country][product_name] = 0
                    country_product_map[record.customer_country][product_name] += 1
                
                # Track monthly counts for trend analysis
                if record.invoice_date:
                    month_key = f"{record.invoice_date.year}-{record.invoice_date.month:02d}"
                    if month_key not in product_data[product_name]['monthly_counts']:
                        product_data[product_name]['monthly_counts'][month_key] = 0
                    product_data[product_name]['monthly_counts'][month_key] += 1
        
        # Calculate trends for each product
        trending_products = []
        for product_name, data in product_data.items():
            # Simple trend: compare first half vs second half of period
            monthly_counts = sorted(data['monthly_counts'].items())
            if len(monthly_counts) >= 2:
                mid = len(monthly_counts) // 2
                first_half_avg = sum(c for _, c in monthly_counts[:mid]) / mid
                second_half_avg = sum(c for _, c in monthly_counts[mid:]) / (len(monthly_counts) - mid)
                
                if second_half_avg > first_half_avg * 1.2:
                    trend = "increasing"
                    growth = ((second_half_avg - first_half_avg) / first_half_avg) * 100
                elif second_half_avg < first_half_avg * 0.8:
                    trend = "decreasing"
                    growth = ((second_half_avg - first_half_avg) / first_half_avg) * 100
                else:
                    trend = "stable"
                    growth = 0
            else:
                trend = "insufficient_data"
                growth = 0
            
            # Get top customers for this product
            top_customers = sorted(
                [(cust, count) for cust, products in customer_product_map.items() if product_name in products
                 for count in [products[product_name]]],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            # Get top countries for this product
            top_countries = sorted(
                [(country, count) for country, products in country_product_map.items() if product_name in products
                 for count in [products[product_name]]],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            trending_products.append({
                'name': product_name,
                'trend': trend,
                'monthly_growth': round(growth, 1) if growth else 0,
                'total_quantity': data['total_quantity'],
                'total_revenue': data['total_revenue'],
                'order_count': data['order_count'],
                'customer_count': len(data['customers']),
                'top_customers': [cust for cust, _ in top_customers],
                'top_countries': [country for country, _ in top_countries]
            })
        
        # Sort by order count (most popular first)
        trending_products = sorted(trending_products, key=lambda x: x['order_count'], reverse=True)[:20]
        
        # Customer preferences (top products per customer)
        customer_preferences = []
        for customer_name, products in customer_product_map.items():
            if not products:
                continue
            
            # Get top 3 products for this customer
            top_products = sorted(products.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Calculate purchase frequency (rough estimate)
            total_purchases = sum(products.values())
            if total_purchases >= 10:
                frequency = "frequent"
            elif total_purchases >= 5:
                frequency = "monthly"
            elif total_purchases >= 2:
                frequency = "occasional"
            else:
                frequency = "rare"
            
            customer_preferences.append({
                'customer_name': customer_name,
                'favorite_products': [prod for prod, _ in top_products],
                'purchase_frequency': frequency,
                'total_purchases': total_purchases
            })
        
        # Sort by total purchases
        customer_preferences = sorted(customer_preferences, key=lambda x: x['total_purchases'], reverse=True)[:15]
        
        return {
            "trending_products": trending_products,
            "customer_preferences": customer_preferences,
            "period_days": days
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch product demand analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Product demand analysis failed: {str(e)}"
        )


@router.get("/dashboard-data-stats")
async def get_dashboard_data_stats(
    current_user: ZodiacUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Counts for the Invoices "Dashboard data" tab: successful validated, converted, and BI records.
    Used so users can see how much data is available and trigger extraction for the Business tab.
    """
    try:
        validated_success = db.query(func.count(InvoiceV2Validated.id)).join(
            InvoiceV2Document, InvoiceV2Validated.document_id == InvoiceV2Document.id
        ).filter(
            InvoiceV2Document.user_id == current_user.id,
            InvoiceV2Document.deleted_at.is_(None),
            InvoiceV2Validated.status == "success"
        ).scalar() or 0
        converted_count = db.query(func.count(ConvertedInvoice.id)).join(
            InvoiceV2Validated, ConvertedInvoice.validated_invoice_id == InvoiceV2Validated.id
        ).join(InvoiceV2Document, InvoiceV2Validated.document_id == InvoiceV2Document.id).filter(
            InvoiceV2Document.user_id == current_user.id
        ).scalar() or 0
        bi_count = db.query(func.count(InvoiceV2BusinessData.id)).filter(
            InvoiceV2BusinessData.user_id == current_user.id
        ).scalar() or 0
        return {
            "validated_success_count": validated_success,
            "converted_count": converted_count,
            "bi_extracted_count": bi_count,
        }
    except Exception as e:
        logger.error(f"Dashboard data stats: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


def _run_backfill_invoice_v2_bi(db: Session, current_user: ZodiacUser, max_invoices: int = 2000) -> tuple:
    """
    Backfill InvoiceV2BusinessData from successful validated invoices (line_items -> products).
    Returns (processed, skipped, errors, total).
    """
    bi_service = InvoiceV2BusinessIntelligence()
    validated_invoices = db.query(InvoiceV2Validated).join(
        InvoiceV2Document,
        InvoiceV2Validated.document_id == InvoiceV2Document.id
    ).filter(
        InvoiceV2Document.user_id == current_user.id,
        InvoiceV2Validated.status == "success"
    ).limit(max_invoices).all()

    processed = skipped = errors = 0
    for validated_invoice in validated_invoices:
        try:
            existing = db.query(InvoiceV2BusinessData).filter(
                InvoiceV2BusinessData.validated_invoice_id == validated_invoice.id
            ).first()
            if existing:
                skipped += 1
                continue
            bi_data = bi_service.extract_bi_data(validated_invoice)
            user_id = bi_data.get("user_id") or current_user.id
            bi_record = InvoiceV2BusinessData(
                validated_invoice_id=bi_data["validated_invoice_id"],
                user_id=user_id,
                customer_id=bi_data["customer"].get("id"),
                customer_name=bi_data["customer"].get("name"),
                customer_country=bi_data["customer"].get("country"),
                supplier_id=bi_data["supplier"].get("id"),
                supplier_name=bi_data["supplier"].get("name"),
                products=bi_data["products"],
                total_products_count=bi_data["total_products_count"],
                total_amount=bi_data["financial"].get("total_amount"),
                tax_amount=bi_data["financial"].get("tax_amount"),
                currency=bi_data["financial"].get("currency"),
                industry=bi_data["industry"],
                industry_confidence=bi_data["industry_confidence"],
                industry_keywords_matched=bi_data["industry_keywords_matched"],
                invoice_date=bi_data["temporal"].get("invoice_date"),
                fiscal_quarter=bi_data["temporal"].get("fiscal_quarter"),
                fiscal_year=bi_data["temporal"].get("fiscal_year"),
                season=bi_data["temporal"].get("season"),
                current_stage=bi_data["lifecycle"].get("current_stage"),
                stage_status=bi_data["lifecycle"].get("stage_status"),
            )
            db.add(bi_record)
            processed += 1
            if processed % 50 == 0:
                db.commit()
        except Exception as e:
            logger.warning(f"Backfill BI invoice {getattr(validated_invoice, 'id', '?')}: {e}")
            errors += 1
    if processed > 0 or errors > 0:
        db.commit()
    return (processed, skipped, errors, len(validated_invoices))


@router.post("/backfill-invoice-v2-bi")
async def backfill_invoice_v2_bi(
    current_user: ZodiacUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Backfill business intelligence data from existing Invoice V2 validated invoices.
    Uses line_items from each successful validated invoice to populate products for the Business tab.
    """
    logger.info(f"🔄 Starting Invoice V2 BI backfill for user {current_user.id}")
    try:
        processed, skipped, errors, total = _run_backfill_invoice_v2_bi(db, current_user)
        logger.info(f"✅ Backfill completed: {processed} processed, {skipped} skipped, {errors} errors")
        return {
            "success": True,
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
            "total": total,
            "message": f"Successfully backfilled BI data for {processed} invoices (products from line_items)",
        }
    except Exception as e:
        logger.error(f"❌ Backfill failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backfill failed: {str(e)}"
        )

