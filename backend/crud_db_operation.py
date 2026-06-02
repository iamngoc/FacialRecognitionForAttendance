"""
C = create
R = read
U = update
D = delete
Write, read, update, delete datas in database
"""

from datetime import date, datetime, timedelta
from typing import Optional, Type
import logging
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.database_model import Employee, TimeRecording

logger = logging.getLogger(__name__)

"""
Employee operations (database "Employees")
"""

def fetch_all_employees(db: Session) -> list[Type[Employee]]:
    """
    fetch all active employees from database
    :param db:
    :return:
    """
    return (db.query(Employee)
            .filter(Employee.active == True)
            .all())

def find_employee_from_id(db: Session, id: int) -> Optional[Employee]:
    """
    find one employee from database from given id
    :param db:
    :param id:
    :return:
    """
    return (db.query(Employee)
            .filter(Employee.id == id)
            .first())

def find_employee_from_email(db: Session, email: str) -> Optional[Employee]:
    """
    find one employee from given email
    :param db:
    :param email:
    :return:
    """
    return (db.query(Employee).
            filter(Employee.email == email)
            .first())

def fetch_embedding_all_employees(db: Session) -> list[dict]:
    """
    fetch all embeddings for facial recognition
    :param db:
    :return:
    """
    employees = fetch_all_employees(db)
    results = []

    for employee in employees:
        if employee.embedding is not None:
            results.append({"id": employee.id,
                            "name": employee.name,
                            "embedding": employee.embedding})

    logger.debug(f"Embeddings loaded: {len(results)} employees")
    return results

def store_embedding(db: Session, employee_id: int, embedding: np.ndarray,
                    photo_path: str = None) -> bool:
    """
    store facial fingerprint embedding of employee and will be called when enrollment
    :param db:
    :param employee_id:
    :param embedding:
    :param photo_path:
    :return:
    """
    employee = find_employee_from_id(db, employee_id)
    if not employee:
        logger.error(f"employee {employee_id} not found")
        return False

    employee.embedding = embedding.tolist() # numpy -> python list
    if photo_path:
        employee.photo_path = photo_path
    employee.updated_in = datetime.now()

    db.commit()
    logger.info(f"Embedding for {employee.name} saved.")
    return True

"""
TimeRecording operations (database "TimeRecording")
"""

def determine_scan_typ(db: Session, employee_id: int) -> str:
    """
    determine whether the next scan type is COME or GO
    if already COME then GO, otherwise COME
    :param db:
    :param employee_id:
    :return: string
    """
    today = date.today()
    # latest scan for today
    latest_scan = (db.query(TimeRecording)
                   .filter(TimeRecording.employee_id == employee_id,TimeRecording.scan_date==today)
                   .order_by(TimeRecording.scan_time)
                   .first())

    if latest_scan is None:
        return f"COME" # not scan yet
    elif latest_scan.scan_type == "COME":
        return f"GO" # already COME, now GO
    else:
        return f"COME" # already DO, now COME again

def enter_scan (db: Session, employee_id: int, scan_type: str, confidence_score: float, camera_id: str ="entrance_00", snapshot_path = None) -> TimeRecording:
    """
    enter a new scan in the database TimeRecording
    :param db: database session
    :param employee_id: id of the employee
    :param scan_type: come or go
    :param confidence_score: check confidence score
    :param camera_id: id of the camera
    :param snapshot_path: path of the snapshot
    :return: information of the new scan
    """

    now = datetime.now()
    entry = TimeRecording(employee_id=employee_id,
                          scan_time=now,
                          scan_date=now.date(),
                          scan_type=scan_type,
                          confidence_score=confidence_score,
                          camera_id=camera_id,
                          snapshot_path=snapshot_path)

    db.add(entry)
    db.commit()
    db.refresh(entry)

    logger.info(f"TimeRecording entered: "
                f"Employee: {employee_id} ,"
                f"{scan_type}, {now.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"Score: {confidence_score} ")

    return entry

def today_report (db: Session) -> list[dict]:
    """all scan for today as report"""
    today = date.today()
    scans = (
        db.query(TimeRecording, Employee)
        .join(Employee)
        .filter(TimeRecording.scan_date == today)
        .order_by(TimeRecording.scan_time)
        .all()
    )

    return [
        {"name": e.name,
             "scan_type": t.scan_type,
             "confidence_score": round(t.confidence_score, 3),
             "camera_id": t.camera_id
        }
        for t, e in scans
    ]

def weekly_report (db: Session, employee_id: int) -> list[dict]:
    """scan report of one employee in the last 7 days"""
    seven_days_ago = datetime.today() - timedelta(days=7)
    scans = (
        db.query(TimeRecording)
        .filter(
            TimeRecording.employee_id == employee_id, TimeRecording.scan_date >= seven_days_ago)
        .order_by(desc(TimeRecording.scan_date), TimeRecording.scan_time)
        .all()
    )

    return [
        {
            "date": str(t.scan_date),
            "scan_type": t.scan_type,
            "time": t.scan_time.strftime("%Y-%m-%d %H:%M:%S"),
            "score": t.confidence_score
        }
        for t in scans
    ]