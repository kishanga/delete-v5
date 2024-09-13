import streamlit as st
import numpy as np
from ultralytics import YOLO
import cv2
import os
import pandas as pd
from collections import defaultdict
import pickle
from pycaret.classification import load_model

# Load YOLOv8 model
yolo_model = YOLO("yolov8m-pose.pt") 

# Prepare the keypoints' names and indices
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Sub-pose ranges in seconds
SUB_POSES = {
    'Sub-pose 1': (0, 3),
    'Sub-pose 2': (3, 8),
    'Sub-pose 3': (8, 12),
    'Sub-pose 4': (12, 22),
    'Sub-pose 5': (22, 30),
    'Sub-pose 6': (30, 35),
    'Sub-pose 7': (35, 37),
    'Sub-pose 8': (37, 40),
    'Sub-pose 9': (40, 46),
    'Sub-pose 10': (46, 50),
    'Sub-pose 11': (50, 55),
    'Sub-pose 12': (55, 60)
}

# Function to map frame index to sub-pose
def get_sub_pose(frame_index):
    seconds = (frame_index / 30)  # Assuming 30 fps
    for sub_pose, (start, end) in SUB_POSES.items():
        if start <= seconds < end:
            return sub_pose
    return None

# Function for processing video and returning keypoints in a Pandas DataFrame
def create_df_coords(video_file):
    rows_list = []
    frame_index = 1
    cap = cv2.VideoCapture(video_file)

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            if frame_index % 60 == 0 and frame_index <= 1800:
                results = yolo_model.track(frame, conf=0.5, stream=True)
                track_history = defaultdict(lambda: [])

                for r in results:
                    boxes = r[0].boxes.xywh.cpu()
                    track_ids = r[0].boxes.id.int().cpu().tolist()

                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))
                        if len(track) > 30:
                            track.pop(0)

                    row = r.keypoints.xyn.cpu().numpy()[0].flatten().tolist()
                    row.insert(0, track_id)
                    row.append(get_sub_pose(frame_index))  # Append sub-pose
                    rows_list.append(row)
        else:
            break
        frame_index += 1
    
    columns = ['track_id']
    for i in range(1, 31):
        columns.append(f'{i}_person')
        for key, _ in KEYPOINT_DICT.items():
            columns.extend([f'{i}_{key}_x', f'{i}_{key}_y'])
    columns.append('sub_pose')  # Add sub-pose column

    keypoints_df = pd.DataFrame(rows_list, columns=columns)
    cap.release()
    cv2.destroyAllWindows()

    return keypoints_df

# Streamlit app
st.title("Video Upload, Keypoints Extraction, and Sub-Pose Display App")

# Video upload
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file:
    temp_file_path = os.path.join("", video_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(video_file.getbuffer())

    # Process the video and get keypoints in DataFrame
    keypoints_df = create_df_coords(temp_file_path)

    # Remove the temporary video file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    # Drop columns containing 'person', 'nose', 'eye', or 'ear'
    columns_to_drop = keypoints_df.filter(regex='person|nose|eye|ear').columns
    keypoints_df = keypoints_df.drop(columns=columns_to_drop)
    keypoints_df.fillna(0, inplace=True)

    # Display the DataFrame with sub-poses
    st.write("Processed Keypoints with Sub-Poses:")
    st.dataframe(keypoints_df)

    # Make predictions using the PyCaret model
    model = load_model('model')
    predictions = model.predict(keypoints_df)

    st.write("Predicted Yoga Form:", predictions[0])
