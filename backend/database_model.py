"""
Sleepy Durian World - SQLAlchemy database model
Define the python-classes, which represent the database tables

SQLAlchemy: translate python and database
"""

from datetime import datetime, date

from sqlalchemy import (Column, Integer, String, Float, Boolean,
                        Date, DateTime, ForeignKey, CheckConstraint)
from sqlalchemy.orm import relationship, declarative_base
from pgvector.sqlalchemy import Vector

base = declarative_base()

class Employee(base):
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(200), unique=True, nullable=False)
    #phone_number = Column(String(20), unique=True, nullable=True)
    department = Column(String(100), default="IT")
    position_ = Column(String(100), default="UA")
    date_of_birth = Column(Date, nullable=True)
    entry_date = Column(Date, default=date.today())

    embedding = Column(Vector(512), nullable=True)
    photo_path = Column(String(500), nullable=True)
    active = Column(Boolean, nullable=True)
    created_in= Column(DateTime, default=datetime.now)
    updated_in= Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # link to Time Recording
    enter_time = relationship("TimeRecording", backref="employees")

    def __repr__(self):
        return f"<Employee id = {self.id}, name = ('{self.name}'>"

class TimeRecording(base):
    __tablename__ = "timerecordings"
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False)
    scan_time = Column(DateTime, default=datetime.now(), nullable=False)
    scan_date = Column(Date, default=date.today(), nullable=False)
    scan_type = Column(String(20), nullable=False) #come/go
    camera_id = Column(String(100),default="entrance_00")
    confidence_score = Column(Float, nullable=False)
    snapshot_path = Column(String(500), nullable=True)

    # relations back to employees
    employee = relationship("Employee", back_populates="enter_time")

    # constraint: only allow to come or go
    __table_args__ = (
        CheckConstraint("scan_typ IN ('COME', ('GO')", name="chk_scan_typ"),
    )

    def __repr__(self):
        return (f"<TimeRecording id={self.id} "
                f"employee_id={self.employee_id} "
                f"type={self.scan_type} time={self.scan_time} ")
