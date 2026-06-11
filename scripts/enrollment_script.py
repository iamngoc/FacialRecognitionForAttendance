"""
Sleepy Durian World - Enrollment Script (Windows Webcam)

This script registers an employee with these steps:
- open the camera (webcam)
- take x photos of the employee (x = 10 or more depends on number you want)
- compute the average embeddings
- store the embeddings into the database 'employees'

Execute: python enrollment_script.py --id 1 --name "Durian Lord"
Execute (folder mode): python enrollment_script.py --id 1 --name "Durian Lord" --folder "C:/photos"
"""
import argparse
import os
import logging
import requests

import cv2
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("enrollment")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

number_photos = 9

def camera_enrollment(employee_id: int, name: str, number_of_photos=number_photos):
    """
    Open the camera and take 'number_photos'.
    User will get instructions for taking photos.
    :param employee_id: id of the employee
    :param name: name of the employee
    :param number_of_photos: number of photos to take
    :return: True if successful, False otherwise
    """
    logger.info(f"Start enrollment for: {name} with id = {employee_id} ...")
    logger.info(f"It will take {number_of_photos} photos.")
    logger.info(f"Please press SPACE for each photo, DELETE to cancel.")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        logger.error("Camera not found.")
        return False

    photos = []
    instructions = [
        "Look directly into the camera",
        "Look slightly to the left",
        "Look slightly to the right",
        "Look slightly upward",
        "Look slightly downward",
        "Smile",
        "Look neutrally",
        "Smile again",
        "Last one, look directly into the camera again",
    ]

    photo_nr = 0
    while photo_nr < number_of_photos:
        ok, frame = camera.read()
        if not ok:
            break

        instruction = (instructions[photo_nr] if photo_nr < len(instructions)
                       else "Press SPACE to take photo")

        h, w = frame.shape[:2]
        cv2.putText(frame, f"Photo {photo_nr + 1}/{number_of_photos}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, instruction,
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "SPACE = Photo --- DEL = Cancel",
                    (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow(f"Enrollment of {name}", frame)
        taste = cv2.waitKey(1) & 0xFF

        if taste == 3014656:  # DEL key
            logger.info("Enrollment cancelled!")
            break
        elif taste == 32:  # SPACE key
            photos.append(frame.copy())
            photo_nr += 1
            logger.info(f"Photo {photo_nr}/{number_of_photos} taken")  # ✅ Fixed: no +1 here

            cv2.putText(frame, "Photo taken successfully!",
                        (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 0), 2)
            cv2.imshow(f"Enrollment of {name}", frame)
            cv2.waitKey(500)

    camera.release()
    cv2.destroyAllWindows()

    if not photos:
        logger.error("No photos were taken.")
        return False

    logger.info(f"Sending {len(photos)} photos to backend ...")
    return send_photos(employee_id, photos, name)


def send_photos(employee_id: int, photos: list, name: str) -> bool:
    """
    Send all photos to /enrollment endpoint.
    :param employee_id:
    :param photos:
    :param name:
    :return: True if successful, False otherwise
    """
    try:
        files = []
        for i, p in enumerate(photos):
            ok, buffer = cv2.imencode(".jpg", p, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            if not ok:
                continue
            files.append(("images", (f"photo_{i:02d}.jpg", buffer.tobytes(), "image/jpeg")))

        url = f"{BACKEND_URL}/enrollment/{employee_id}"
        response = requests.post(url, files=files, timeout=60)

        if response.status_code == 200:
            datas = response.json()
            mes = datas["message"]
            handled = datas["handling_images"]
            missing = datas["missing_images"]
            logger.info("Successful!")
            logger.info(f"{mes}")
            logger.info(f"Handled images: {handled}")
            if datas.get("missing_images"):
                logger.warning(f"Error: {missing}")
            return True
        else:
            logger.error(f"Error: {response.status_code}")
            logger.error(f"{response.text}")
            return False

    except requests.exceptions.ConnectionError:
        logger.error(f"Backend not available at {BACKEND_URL}")
        logger.error("Is the system running? (docker-compose up)")
        return False


def folder_enrollment(employee_id: int, name: str, folder_path: str) -> bool:
    """
    Alternative: load photos from folder instead of using webcam.
    :param employee_id:
    :param name:
    :param folder_path:
    :return: True if successful, False otherwise
    """
    photo_paths = (glob.glob(f"{folder_path}/*.jpg") +
                   glob.glob(f"{folder_path}/*.png") +
                   glob.glob(f"{folder_path}/*.jpeg"))

    if not photo_paths:
        logger.error(f"No photos found in folder: {folder_path}")
        return False

    logger.info(f"Found {len(photo_paths)} photos in folder: {folder_path}")
    photos = []
    for path in photo_paths:
        img = cv2.imread(path)
        if img is not None:
            photos.append(img)

    return send_photos(employee_id, photos, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Employee Enrollment for Sleepy Durian World")

    # required=True is valid for optional arguments
    parser.add_argument("--id",     type=int, required=True,  help="Employee ID (e.g. 1)")
    parser.add_argument("--name",   type=str, required=True,  help="Employee name (e.g. 'Slepy Durian')")
    # --folder is optional (only needed when not using webcam)
    parser.add_argument("--folder", type=str, required=False, default=None,
                        help="Folder with photos instead of webcam (optional)")

    args = parser.parse_args()

    if args.folder:
        # Folder mode
        success = folder_enrollment(args.id, args.name, args.folder)
    else:
        # Webcam mode (default)
        success = camera_enrollment(args.id, args.name)

    if success:
        logger.info(f"Enrollment for '{args.name}' completed successfully!")
    else:
        logger.error(f"Enrollment for '{args.name}' failed.")
