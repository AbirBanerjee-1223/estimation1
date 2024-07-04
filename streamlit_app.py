import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import tempfile
import boto3
import os
from dotenv import load_dotenv

# AWS S3 credentials (please replace these with actual credentials)
load_dotenv()
access_key = os.getenv("AWS_ACCESS_KEY")
secret_key = os.getenv("AWS_SECRET_KEY")
bucket_name = os.getenv("AWS_BUCKET_NAME")
region = os.getenv("AWS_REGION")
s3_path = "combined_csv_abir/combined_angles_final.csv"

# Function to download the CSV file from S3
def download_from_s3(access_key, secret_key, bucket_name, region, s3_path, local_path):
    s3 = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    s3.download_file(bucket_name, s3_path, local_path)

# Angle calculation function
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# Function to add text to each frame
def add_text(image, prediction):
    cv2.putText(image, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
    return image

# Function to calculate average angles for a segment
def calculate_average_angles(segment_frames, angle_pairs):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    angles_list = []
    for frame in segment_frames:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            angles = []
            for pair in angle_pairs:
                a = [landmarks[mp_pose.PoseLandmark[pair[0]].value].x, landmarks[mp_pose.PoseLandmark[pair[0]].value].y]
                b = [landmarks[mp_pose.PoseLandmark[pair[1]].value].x, landmarks[mp_pose.PoseLandmark[pair[1]].value].y]
                c = [landmarks[mp_pose.PoseLandmark[pair[2]].value].x, landmarks[mp_pose.PoseLandmark[pair[2]].value].y]
                angles.append(calculate_angle(a, b, c))
            angles_list.append(angles)
    
    if not angles_list:
        return None
    
    return np.mean(angles_list, axis=0)

# Download the CSV file from S3
local_csv_path = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name
download_from_s3(access_key, secret_key, bucket_name, region, s3_path, local_csv_path)

# Load the data and train the model
data = pd.read_csv(local_csv_path)
X = data.drop(columns=["Frame", "label"])
y = data["label"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

angle_pairs = [
    ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
    ('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),
    ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'),
    ('LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'),
    ('RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'),
    ('LEFT_ELBOW', 'LEFT_SHOULDER', 'LEFT_HIP'),
    ('RIGHT_ELBOW', 'RIGHT_SHOULDER', 'RIGHT_HIP'),
    ('RIGHT_ELBOW','RIGHT_WRIST','RIGHT_INDEX'),
    ('LEFT_ELBOW','LEFT_WRIST','LEFT_INDEX')
]

# Streamlit app
st.title("Exercise Prediction from Video")
uploaded_video = st.file_uploader("Upload a video", type=["mp4"])
if uploaded_video is not None:
    video_bytes = uploaded_video.read()
    temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    with open(temp_file_path, 'wb') as f:
        f.write(video_bytes)
    
    cap = cv2.VideoCapture(temp_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    segment_length = 30
    segment_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        segment_frames.append(frame)
        if len(segment_frames) == segment_length:
            avg_angles = calculate_average_angles(segment_frames, angle_pairs)
            if avg_angles is not None:
                prediction = model.predict([avg_angles])[0]
            else:
                prediction = "Unknown"
            for frame in segment_frames:
                annotated_frame = add_text(frame, prediction)
                out.write(annotated_frame)
            segment_frames = []
    
    if segment_frames:
        avg_angles = calculate_average_angles(segment_frames, angle_pairs)
        if avg_angles is not None:
            prediction = model.predict([avg_angles])[0]
        else:
            prediction = "Unknown"
        for frame in segment_frames:
            annotated_frame = add_text(frame, prediction)
            out.write(annotated_frame)
    
    cap.release()
    out.release()
    
    st.video(output_path)
