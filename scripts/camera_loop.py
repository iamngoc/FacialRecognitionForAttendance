"""
Run this script to take photos.
The photos will be sent to backend
"""
from __future__ import annotations

import os
import logging
import sys

import requests
import time
from datetime import datetime

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logger = logging.getLogger("camera")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
SCAN_INTERVAL = 2 # seconds among scan
DISPLAY_INTERVAL = 4 # second to display the greetings

def send_frame(frame:np.ndarray) -> dict or None:
    """send an image to backend to do facial recognition"""
    try:
        ok, buffer = cv2.imencode('.jpg', [frame, cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return None

        response = requests.post(f"{BACKEND_URL}/scan",
                               files={"image": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                               data={"camera_id": "entrance_00"},
                               timeout=10)
        if response.status_code == 200:
            return response.json()
        return None

    except requests.exceptions.ConnectionError:
        logger.warning("Backend not available")
        return None
    except Exception as e:
        logger.error(f"Error at sending: {e}")
        return None

def greeting_overlay(frame:np.ndarray, message: str, recognized: bool) -> np.ndarray:
    """draw a beautiful greeting overlay"""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # background box
    color = (0, 200, 0) if recognized else (0, 0, 200)
    cv2.rectangle(overlay, (0, h-80), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.4, 0, frame)

    # message
    cv2.putText(frame, message,
                (10, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame

def main():
    """camera loop"""
    logger.info("Starting camera loop ...")
    logger.info(f"Backend URL: {BACKEND_URL}")
    logger.info("Press Ctrl+C to exit")

    # check backend connection
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        logger.info("Backend connected!")
    except Exception:
        logger.error(f"Backend not available, Backend URL {BACKEND_URL}")
        logger.error("Please start docker-compose first with: docker-compose up")
        sys.exit(1)

    # open the camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        logger.error("camera not found!")
        sys.exit(1)

    # set up camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_scan = 0
    last_response = None
    display_till = 0

    logger.info("Camera started successfully! System is running!")

    while True:
        ok, frame = camera.read()
        if not ok:
            logger.warning("No image.")
            time.sleep(1)
            continue

        now = time.time()
        display_frame = frame.copy()

        # send every SCAN_INTERVALL seconds one scan
        if (now - last_scan) >= SCAN_INTERVAL:
            last_scan = now
            result = send_frame(frame)

            if result and result.get("recognized"):
                name = result["name"]
                scan_type = result.get("scan_type")
                confidence = result.get("confidence")
                last_response = result
                display_till = now + DISPLAY_INTERVAL
                logger.info(f"{name} - {scan_type} with {confidence} confidence")

        # show greeting when still current
        if last_response and now < display_till:
            message = last_response.get("message", "")
            display_frame = greeting_overlay(display_frame, message, True)

        # time and status
        time_now = datetime.now().strftime("%I:%M:%S")
        cv2.putText(display_frame, f"{time_now}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Facial Recognition from SleepyDurian - 'Ctrl+C' to exit", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('Ctrl+C'):
            break

    camera.release()
    cv2.destroyAllWindows()
    logger.info("Camera stopped successfully! See you!")

if __name__ == "__main__":
    main()



