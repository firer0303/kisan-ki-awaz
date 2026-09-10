"""
Kisan Ki Awaz - Database / Session Service
============================================
Manages farmer session history and analysis records
using SQLite via SQLAlchemy.
"""
import json
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from config import settings

Base = declarative_base()


class AnalysisRecord(Base):
    """Database model for a single farmer analysis."""
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True)
    language = Column(String(10))
    input_type = Column(String(20))  # "voice", "image", "camera"
    farmer_question = Column(Text)
    crop_detected = Column(String(50), nullable=True)
    disease_detected = Column(String(100), nullable=True)
    confidence = Column(Float, default=0.0)
    risk_level = Column(String(20))
    response_text = Column(Text)
    sources_used = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseService:
    """Manages database operations for session history."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or settings.database.url
        self.engine = create_engine(self.db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info(f"Database service initialized ({self.db_url})")

    def save_analysis(self, data: Dict) -> int:
        """Save an analysis record. Returns the record ID."""
        session: Session = self.SessionLocal()
        try:
            record = AnalysisRecord(
                session_id=data.get("session_id", ""),
                language=data.get("language", "en"),
                input_type=data.get("input_type", "unknown"),
                farmer_question=data.get("farmer_question", ""),
                crop_detected=data.get("crop_detected"),
                disease_detected=data.get("disease_detected"),
                confidence=data.get("confidence", 0),
                risk_level=data.get("risk_level", "unknown"),
                response_text=data.get("response_text", ""),
                sources_used=json.dumps(data.get("sources_used", [])),
            )
            session.add(record)
            session.commit()
            record_id = record.id
            logger.info(f"Analysis record saved (ID: {record_id})")
            return record_id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save analysis: {e}")
            return -1
        finally:
            session.close()

    def get_session_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        """Retrieve analysis history for a session."""
        session: Session = self.SessionLocal()
        try:
            records = (
                session.query(AnalysisRecord)
                .filter(AnalysisRecord.session_id == session_id)
                .order_by(AnalysisRecord.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "language": r.language,
                    "input_type": r.input_type,
                    "farmer_question": r.farmer_question,
                    "crop_detected": r.crop_detected,
                    "disease_detected": r.disease_detected,
                    "confidence": r.confidence,
                    "risk_level": r.risk_level,
                    "response_text": r.response_text[:200] + "..." if len(r.response_text) > 200 else r.response_text,
                    "sources_used": json.loads(r.sources_used) if r.sources_used else [],
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in records
            ]
        except Exception as e:
            logger.error(f"Failed to retrieve history: {e}")
            return []
        finally:
            session.close()

    def get_stats(self) -> Dict:
        """Get overall database statistics."""
        session: Session = self.SessionLocal()
        try:
            total = session.query(AnalysisRecord).count()
            return {
                "total_analyses": total,
                "unique_sessions": session.query(AnalysisRecord.session_id).distinct().count(),
                "languages_used": [
                    r[0] for r in session.query(AnalysisRecord.language).distinct().all()
                ],
            }
        except Exception as e:
            logger.error(f"Stats query failed: {e}")
            return {"total_analyses": 0, "unique_sessions": 0, "languages_used": []}
        finally:
            session.close()
