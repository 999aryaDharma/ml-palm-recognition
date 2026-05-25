from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, LargeBinary, Text
from sqlalchemy.orm import relationship
from datetime import datetime
import json

from db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False, index=True)
    enrolled_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    templates = relationship("Template", back_populates="user", cascade="all, delete-orphan")
    demo_logs = relationship("DemoLog", back_populates="user")


class Template(Base):
    __tablename__ = "templates"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    embedding = Column(LargeBinary, nullable=False)
    quality_score = Column(Float, default=0.0, nullable=False)
    captured_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="templates")


class DemoLog(Base):
    __tablename__ = "demo_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    demo_type = Column(String(40), nullable=False, index=True)
    payload_json = Column(Text, default="{}", nullable=False)
    match_score = Column(Float, default=0.0, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="demo_logs")

    @property
    def payload(self):
        return json.loads(self.payload_json or "{}")
