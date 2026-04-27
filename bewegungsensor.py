"""
This only works on Linux
Needed Hardware:

1. Raspberry Pi
2. USB-Webcam, which is compatible with Raspberry Pi
3. Moving Sensor
4. Micro-SD Card: for saving memory of photos
5. Power supply and maybe another connections
"""

# Moving Sensor configuration
# A photo is made wenn a movement detected
# Test before use

"""
from gpiozero import MotionSensor
from picamera2 import PiCamera2
from time import sleep
from signal import pause

# a motion sensor and the PiCamera
pir = MotionSensor(4)
camera = PiCamera()

# start the camera
camera.start_preview()

# image names
i = 0

# take photo when motion is detected:
def take_photo():
    global i
    i = i + 1
    camera.capture('/home/pi/Desktop/photo_%s.jpg' % i)
    print("A Photo has been taken!")
    sleep(1) # sleep 1 second

# assign a function that runs when motion is detected
pir.when_motion = take_photo

pause()
"""



