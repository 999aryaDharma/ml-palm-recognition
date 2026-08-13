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
    demo_logs = relationship("DemoLog", back_populates="user", cascade="all, delete-orphan")
    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    wallet = relationship("Wallet", back_populates="user", uselist=False, cascade="all, delete-orphan")


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True)
    nik = Column(String(50), nullable=True)
    kelas_jabatan = Column(String(100), nullable=True)

    user = relationship("User", back_populates="profile")


class Wallet(Base):
    __tablename__ = "wallets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True)
    balance = Column(Float, default=0.0, nullable=False)

    user = relationship("User", back_populates="wallet")
    transactions = relationship("WalletTransaction", back_populates="wallet", cascade="all, delete-orphan")


class WalletTransaction(Base):
    __tablename__ = "wallet_transactions"

    id = Column(Integer, primary_key=True, index=True)
    wallet_id = Column(Integer, ForeignKey("wallets.id"), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    transaction_type = Column(String(20), nullable=False)
    description = Column(String(255), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    wallet = relationship("Wallet", back_populates="transactions")


class Template(Base):
    __tablename__ = "templates"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    model_id = Column(String(80), nullable=False, default="mobilefacenet-pretrained", index=True)
    model_version = Column(String(20), nullable=False, default="1.0.0", index=True)
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
