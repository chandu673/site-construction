import os
import cv2
import numpy as np
import streamlit as st
import boto3
import torch
from twilio.rest import Client
from PIL import Image
import io
import tempfile
from dotenv import load_dotenv

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Twilio Credentials (from .env)
ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE")
ALERT_PHONE = os.getenv("ALERT_PHONE")

client = Client(ACCOUNT_SID, AUTH_TOKEN)

# ✅ AWS S3 Credentials (from .env)
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name='us-west-2'
)

# ✅ Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)

# ✅ Global variable to track alert
alert_sent = False

def send_alert(message):
    global alert_sent
    if not alert_sent:
        try:
            client.messages.create(
                body=message,
                from_=TWILIO_PHONE,
                to=ALERT_PHONE
            )
            alert_sent = True
        except Exception as e:
            st.warning(f"⚠️ Alert failed: {e}")

def upload_to_s3(image, filename):
    if image.mode == "RGBA":
        image = image.convert("RGB")
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    s3.put_object(Bucket=BUCKET_NAME, Key=filename, Body=img_bytes.getvalue())
    st.success("✅ Uploaded to S3")

def detect_objects(image):
    img_array = np.array(image)
    results = model(img_array)
    detected_classes = results.pandas().xyxy[0]["name"].tolist()

    if "helmet" not in detected_classes or "vest" not in detected_classes:
        send_alert("⚠️ Safety Violation: Missing PPE detected on site!")
        st.error("⚠️ Safety Violation Detected: Missing PPE!")

    return results.render()[0]

def process_video(file):
    global alert_sent
    alert_sent = False

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(file.read())
        tmp_video_path = tmp_video.name

    video_capture = cv2.VideoCapture(tmp_video_path)
    frames = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        processed_frame = detect_objects(img)
        frames.append(processed_frame)

    video_capture.release()

    if not frames:
        st.error("❌ No valid frames to process.")
        return None

    height, width, _ = np.array(frames[0]).shape
    output_path = "processed_video.mp4"
    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height))

    for frame in frames:
        output_video.write(np.array(frame))

    output_video.release()
    os.remove(tmp_video_path)

    with open(output_path, "rb") as f:
        return f.read()

# ✅ Streamlit UI
def main():
    st.title("🏗️ AI-Powered Construction Site Monitoring")
    uploaded_file = st.file_uploader("📤 Upload an image or video", type=["jpg", "png", "jpeg", "mp4", "avi"])

    if uploaded_file:
        if uploaded_file.type.startswith("image"):
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)
            processed_image = detect_objects(image)
            st.image(processed_image, caption="🔍 Detection Result", use_container_width=True)
            upload_to_s3(image, f"construction_site/{uploaded_file.name}")
        
        elif uploaded_file.type.startswith("video"):
            video_bytes = process_video(uploaded_file)
            if video_bytes:
                with open("temp_processed_video.mp4", "wb") as f:
                    f.write(video_bytes)
                st.video("temp_processed_video.mp4")

if __name__ == "__main__":
    main()

