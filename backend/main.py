"""
FastAPI-Web server
it makes Endpoint (URLs) available
Methods:
    GET /                       -> system status
    POST /scan                  -> facial recognition from camera
    POST /enrollment/{employee_id}       -> employee enrollment (load photo)
    GET /report/today           -> to display all scans of today
    GET /report/week/{employee_id}       -> to display all scans of week of determined employee
    GET /employees              -> to display all employees
    GET /health                 -> review health of Docker

create a connection to backend and give all required information
"""

import os
#import io
import logging
import numpy as np
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

import cv2
# !pip install fastapi==0.125.0
from fastapi import (FastAPI, Depends, HTTPException, File, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
#from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
#from PIL import Image

import crud_db_operation
from database import connect_database, create_table, get_db
from faceEngine import FacialRecognizingEngine

# configurate Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logger = logging.getLogger("backend")

# Configuration from environment variables
THRESHOLD = float(os.getenv("THRESHOLD", "0.45"))
ANTI_SPAM_SEC = int(os.getenv("ANTI_SPAM_SEC", "30"))

# Global AI-Engine
ai_engine: Optional[FacialRecognizingEngine] = None
# Anti spam: [employee_id = last_recognizing_time]
last_scan: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """will be executed at starting and finishing"""
    global ai_engine
    logger.info(f"Lifespan is starting ...")

    # check database connection
    if not connect_database():
        raise RuntimeError("Database connection failed!")
    create_table()

    # load AI model
    logger.info(f"Loading ArcFace AI model ...")
    ai_engine = FacialRecognizingEngine(threshold=THRESHOLD)
    logger.info("System is ready. Facial Recognizing is running.")

    yield # Application runs here

    logger.info("System will be finished ...")

# create FastSPI app
app = FastAPI(
    title="Facial Recognization",
    description="Time Recording automatically via ArcFace",
    version="1.0",
    lifespan=lifespan
)

# Dashboard is able to interact to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model (API format)
class ScanAnswer(BaseModel):
    recognized: bool
    employee_id: Optional[int] = None
    name: Optional[str] = None
    scan_type: Optional[str] = None
    time: Optional[str] = None
    confidence: Optional[float] = None
    message: str

# API Endpoint
@app.get("/")
async def root():
    """system status"""
    return {"system": "Facial Recognizing by SleepyDurian",
            "version": "1.0.0",
            "status": "Ready",
            "ai_engine": "ArcFace (InsightFace buffalo_1)",
            "threshold": THRESHOLD,
            "time": datetime.now().strftime("%m/%d/%Y %I:%M:%S %p")}

@app.get("/employees")
async def employees_list(db: Session = Depends(get_db)):
    """all active employees with embedding status"""
    all_emps = crud_db_operation.fetch_all_employees(db)
    return [
        {"id": e.id,
         "name": e.name,
         "email": e.email,
         "phone_number": e.phone_number,
         "department": e.department,
         "position": e.position,
         "date_of_birth": e.date_of_birth,
         "entry_date": e.entry_date,
         "embedding": e.embedding is not None,
         "active": e.active}
        for e in all_emps
    ]

@app.get("/health")
async def health():
    """Docker health check"""
    return {"status": "ok"}

@app.get("/report/today")
async def report_today(db: Session = Depends(get_db)):
    """display all scans for today"""
    return {"date": datetime.now().strftime("%m/%d/%Y"),
            "scans": crud_db_operation.today_report(db)}

@app.get("/report/week/{employee_id}")
async def report_week(employee_id: int, db: Session = Depends(get_db)):
    """display all scans for the last 7 day of employee_id"""
    employee = crud_db_operation.find_employee_from_id(db, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    return {"date": datetime.now().strftime("%m/%d/%Y"),
            "employee": employee.name,
            "scans": crud_db_operation.weekly_report(db, employee_id)}

@app.post("/scan", response_model=ScanAnswer)
async def scan_face(image: UploadFile = File(...),
                    camera_id: str = "entrance_00",
                    db: Session = Depends(get_db)):
    """
    Take an image of face, recognize it and enter the time of taking image of face.
    :param image:
    :param camera_id:
    :param db:
    :return:
    """
    # load image
    content = await image.read()
    image_array = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Image invalid!")

    # load all embeddings from the database
    embeddings = crud_db_operation.fetch_embedding_all_employees(db)

    if not embeddings:
        return ScanAnswer(
            recognized=False,
            message="No employee detected"
        )

    # facial recognizing
    result = ai_engine.recognize_person(frame, embeddings)

    if result is None:
        return ScanAnswer(
            recognized=False,
            message="Person not recognized. Please try again in the next 30 seconds or register first."
        )
    emp_id = result["employee_id"]
    now = datetime.now()

    # Anti-spam: 30 seconds waiting time for the next san
    if emp_id in last_scan:
        diff = (now - last_scan[emp_id]).total_seconds()
        if diff < ANTI_SPAM_SEC:
            return ScanAnswer(
                recognized=True,
                employee_id=emp_id,
                name=result["name"],
                confidence=result["confidence"],
                message=(f"{result['name']}, please wait."
                         f"You can scan in the next {int(ANTI_SPAM_SEC-diff)} seconds.")
            )

    # determine the scan type
    scan_type = crud_db_operation.determine_scan_typ(db, emp_id)

    # enter scan type in the database TimeRecording
    crud_db_operation.enter_scan(db=db,
                                 employee_id=emp_id,
                                 scan_type=scan_type,
                                 confidence_score=result["confidence"],
                                 camera_id=camera_id)

    last_scan[emp_id] = now
    time = now.strftime("%H:%M:%S")
    name = result["name"]
    message = (f"Hello {name}! Welcome" if scan_type=="COME" else f"See you!"
               f"{scan_type} at {time}")
    return ScanAnswer(
        recognized=True,
        employee_id=emp_id,
        name=result["name"],
        scan_type=scan_type,
        time=time,
        confidence=round(result["confidence"], 3),
        message=message
    )

@app.post("/enrollment/{employee_id}")
async def enroll_employee(employee_id: int,
                          images: list[UploadFile] = File(...),
                          db: Session = Depends(get_db)):
    """
    Enroll a employee with more images. Compute the average embedding from all images.
    -> AI model can recognize the face from different angles and lighting
    :param employee_id:
    :param images:
    :param db:
    :return:
    """
    employee = crud_db_operation.find_employee_from_id(db, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail=f"Employee with {employee_id} not found!")

    all_embeddings = []
    missing_image = []
    for image in images:
        try:
            # read the image file
            content = image.read()
            image_array = np.frombuffer(content, np.uint8)
            i = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if i is None:
                missing_image.append(image.filename)
                continue

            # compute embedding
            embedding = ai_engine.compute_embedding(i)
            if embedding is not None:
                all_embeddings.append(embedding)
            else:
                missing_image.append(f"{image.filename} (no face detected)")

        except Exception as e:
            error = str(e)
            missing_image.append(f"{image.filename} (Error: {error})")

    if not all_embeddings:
        raise HTTPException(status_code=400, detail="No face in loaded images detected!")

    # compute average embedding
    average_embedding = np.mean(all_embeddings, axis=0)

    # store into database
    succeed = crud_db_operation.store_embedding(db, employee_id, average_embedding)
    name = employee.name
    return {
        "success": succeed,
        "name": employee.name,
        "handling_images": len(all_embeddings),
        "message": f"{name} enrolled successfully!, {len(all_embeddings)} handled."
    }

