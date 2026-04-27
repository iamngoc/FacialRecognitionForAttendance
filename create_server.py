from flask import Flask, request
import os

app = Flask(__name__)

upload_folder = "uploads"
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)


@app.route("/upload", methods=["POST"])
def upload_photo():
    file = request.files["photo"]
    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)
    print("Photo uploaded", filepath)
    return "OK"

# Test function
@app.route("/", methods=["GET"])
def hello_world():
    return "Hello World!"

@app.route("/howareyou", methods=["GET"])
def howareyou():
    #print("Hi! How are you today?")
    return "Hi! How are you today?"


app.run(host="0.0.0.0", port=8080)