"""
Movement Detection with OpenCV

Step:
1. Camera continuously takes photos
2. current photo is compared to the previous photo
3. When large change -> movement, then save the photo
4. Build a server for upload photos
"""
import os
import cv2
import time
from flask import Flask, request

app = Flask(__name__)

# create a photo folder, upload folder
photos_folder = f"captured_"
if not os.path.exists(photos_folder):
    os.makedirs(photos_folder)

upload_folder = "uploads"
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

@app.route("/upload", methods=["POST"])
def upload_photo():
    file = request.files[f"[photo]"]
    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)
    print("Photo uploaded", filepath)
    return "OK"

app.run(host="0.0.0.0", port=8080, debug=True)

# start camera ( 0 = first webcam)
capture = cv2.VideoCapture(0)

# a short break, to that the camera will stable
time.sleep(2) # 2second break

# take the first photo as a reference
ret, frame = capture.read()
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

print("Start monitoring now ...")

def make_photo(photo_folder):
    filename = photo_folder + f"photo_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    print("A movement is detected. Photo saved with name:", filename)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    # prepare current photo
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Difference among Frames
    frame_diff = cv2.absdiff(prev_gray, gray)

    # Threshold (make movement visible)
    threshold = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    threshold = cv2.dilate(threshold,None, iterations=2)

    # Find contours (moving areas)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        # small movements will be ignored (Noise)
        if cv2.contourArea(contour) > 5000: # when too much alarm, increase the value and vice versa
            motion_detected = True

    # when movement detected -> save photo
    if motion_detected:
        make_photo(photos_folder)

        # a short break to avoid taking 100 photos
        time.sleep(3)

    # set current photo as new reference
    prev_gray = gray

    # escape
    if cv2.waitKey(1) == 1234:
        break

capture.release()
cv2.destroyAllWindows()

