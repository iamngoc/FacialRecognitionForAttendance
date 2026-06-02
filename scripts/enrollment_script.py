"""
Sleepy Durian World - Enrollment Script (Windows Webcam)

This script register an employee with these steps:
- open the camera (webcam)
- take x photos of the employee (x = 10 or more depends on number you want)
- compute the average embeddings
- store the embeddings into the database 'employees'

Execute: enrollment_script.py --id 01 --name -- Durian Lord
"""
import argparse
import os
import logging
import requests

import cv2
import glob

logging.basicConfig(level=logging.info(), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("enrollment")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

number_photos = 9

def camera_enrollment(employee_id: int, name: str, number_of_photos = number_photos):
    """
    open the camera and take 'number_photos'.
    User will get the instruction of taking photos
    :param employee_id: id of the employee
    :param name: name of the employee
    :param number_of_photos: number of photos to take
    :return:
    """
    logger.info(f"Start enrollment for: {name} with id = {employee_id} ...")
    logger.info(f"It will take {number_of_photos} photos.")
    logger.info(f"Please press SPACE for each photo, DELETE for cancel.")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        logger.error("Camera not found.")
        return False

    photos = []
    instructions = [
        "Look directly in the camera",
        "Look slightly to the left",
        "Look slightly to the right",
        "Look slightly to the top",
        "Look slightly to the bottom",
        "Smile",
        "Look neutrally",
        "Smile again",
        "The last one, please look directly in the camera again",
    ]

    photo_nr = 0
    while photo_nr < number_of_photos:
        ok, frame = camera.read()
        if not ok:
            break

        # show the instruction
        instruction = (instructions[photo_nr] if photo_nr < len(instructions)
                       else "Please press SPACE to take photos")

        # write text into photo
        h, w = frame.shape[:2]
        cv2.putText(frame, f"photo {photo_nr+1}/{number_of_photos}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, instruction,
                    (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "SPACE = Photo --- DEL = Cancel",
                    (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow(f"Enrollment of {name}", frame)
        taste = cv2.waitKey(1) & 0xFF


        if taste == 3014656: #DEL   # taste == ord('del')
            logger.info(f"Enrollment cancelled!")
            break
        elif taste == 32: #SPACE
            photos.append(frame.copy())
            photo_nr += 1
            logger.info(f"Photo {photo_nr+1}/{number_of_photos} taken")

            # flash briefly as confirmation
            cv2.putText(frame, "Photo taken sucessfully!",
                        (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 0), 2)
            cv2.imshow(f"Enrollment of {name}", frame)
            cv2.waitKey(500)

    camera.release()
    cv2.destroyAllWindows()

    if not photos:
        logger.error("No photos were taken.")
        return False

    # send photos to backend
    logger.info(f"Sending {len(photos)} photos to backend ...")
    return send_photos(employee_id, photos, name)

def send_photos(employee_id: int, photos: list, name: str) -> bool:
    """
    send alls photos to /enrollment endpoint
    :param employee_id: 
    :param photos: 
    :param name: 
    :return: 
    """
    try:
        files = []
        for i, p in enumerate(photos):
            # OpenCV -> JPEG Bytes
            ok, buffer = cv2.imencode(".jpg", p, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            if not ok:
                continue
            files.append(("photos", (f"photo_{i:02d}.jpg", buffer.tobytes(), "image/jpeg")))
            
        url = f"{BACKEND_URL}/enrollment/{employee_id}"
        response = requests.post(url, files=files, timeout=60)
        
        if response.status_code == 200:
            datas = response.json()
            mes = datas["message"]
            handled = datas["handling_images"]
            missing = datas["missing_images"]
            logger.info(f"Successful!")
            logger.info(f"{mes}")
            logger.info(f"Handled images: {handled}")
            if datas.get("missing_images"):
                logger.warning(f"Error: {missing}")
            return True
        else:
            logger.error(f"Error: {response.status_code}")
            logger.error(f"{response.text}")

    except requests.exceptions.ConnectionError:
        logger.error(f"Backend not available. {BACKEND_URL}")
        logger.error("Check whether system was started correctly? (docker-compose up)")
        return False

def folder_enrollment (employee_id: int, name: str, folder_path: str) -> bool:
    """
    alternative: load photos from folder instead of using webcam
    :param employee_id:
    :param name:
    :param folder_path:
    :return:
    """
    photo_paths = (glob.glob(f"{folder_path}/*.jpg") +
                  glob.glob(f"{folder_path}/*.png") +
                  glob.glob(f"{folder_path}/*.jpeg"))

    if not photo_paths:
        logger.error(f"No photos in folder {folder_path} found.")
        return False

    logger.info(f"Found {len(photo_paths)} photos in folder {folder_path}")
    photos = []
    for path in photo_paths:
        img = cv2.imread(path)
        if img is not None:
            photos.append(img)

    return send_photos(employee_id, photos, name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrollment for Employees of SleepyDurian")
    parser.add_argument("__id", type=int, required=True, help="Employee-ID")
    parser.add_argument("__name", type=str, required=True, help="Employee-Name")
    parser.add_argument("__folder", type=str, required=True, help="Folder with photos (no webcam)")
    parser.add_argument("__photos", type=str, default=None, help="Photos to send to backend")


            
            
