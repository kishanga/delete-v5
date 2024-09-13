import streamlit as st
import numpy as np
from ultralytics import YOLO
import cv2
import os
import pandas as pd
from collections import defaultdict
import pickle
from pycaret.classification import load_model

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

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

# Function for processing video and returning keypoints in a Pandas DataFrame
def create_df_coords(video_file):
    # Initialise list to store keypoints of all frames
    rows_list = []

    # Set frame index for tracking frame number
    frame_index = 1

    # Create a VideoCapture object to open the video file
    cap = cv2.VideoCapture(video_file)

    # Loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Extract the landmarks for every 60 frames i.e. 2 seconds
            if frame_index % 60 == 0 and frame_index <= 1800:
                results = yolo_model.track(frame, conf=0.5, stream=True)

                # Store the track history
                track_history = defaultdict(lambda: [])

                # process through results generator
                for r in results:
                    boxes = r[0].boxes.xywh.cpu()
                    track_ids = r[0].boxes.id.int().cpu().tolist()

                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))
                        if len(track) > 30:
                            track.pop(0)

                    # retrieve keypoints, add keypoints to df
                    row = r.keypoints.xyn.cpu().numpy()[0].flatten().tolist()
                    row.insert(0, track_id)

                    # append row to rows_list
                    rows_list.append(row)
        else:
            break

        frame_index += 1
      
    
    
    
    
    # Create column names for data frame used for prediction
    columns = []
    for i in range(1, 31):
        columns.append(str(i) + '_' + "person")
    
        for key, value in KEYPOINT_DICT.items():
            columns.extend([str(i) + '_' + key + '_x', str(i) + '_' + key + '_y']) 
                       
    # Flatten rows_list
    flattened_rows_list = [item for sublist in rows_list for item in sublist]
    
    # Convert to DataFrame with a single row
    keypoints_df = pd.DataFrame([flattened_rows_list], columns=columns)



    

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    return keypoints_df



# Streamlit app
st.title("Subpose Detection")

# Video upload
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# Process video if uploaded
if video_file:
    # Create the temp_video directory if it doesn't exist
    #if not os.path.exists("temp_video"):
    #    os.makedirs("temp_video")

    # Save uploaded file temporarily
    temp_file_path = os.path.join("", video_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(video_file.getbuffer())

    #st.write(f"Video saved to {temp_file_path}")

    # Process the video and get keypoints in DataFrame
    keypoints_df = create_df_coords(temp_file_path)

    #st.success("Keypoints extracted and saved to DataFrame")
    
    # Delete the temporary video file after predictions are made
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
        #st.write(f"Temporary file {video_file.name} deleted.")

    # Dropping columns containing 'person', 'nose', 'eye', or 'ear' in their names
    columns_to_drop = keypoints_df.filter(regex='person|nose|eye|ear').columns
    keypoints_df = keypoints_df.drop(columns=columns_to_drop)

    # Fill the missing keypoints with 0 due to lack of frames in some videos
    keypoints_df.fillna(0, inplace=True)

    # Display the updated DataFrame
    #st.write("Processed Keypoints Data:")
    #st.dataframe(keypoints_df.head())

    ############

    # Find the missing columns by comparing expected with the current DataFrame columns
    missing_columns = set(columns) - set(keypoints_df.columns)
    
    # Add missing columns to the DataFrame, initialized with zero values
    for col in missing_columns:
        keypoints_df[col] = 0
    
    # Ensure the DataFrame columns are in the same order as `expected_columns`
    keypoints_df = keypoints_df[columns]


    ############



    

    # Make predictions using the PyCaret model
    #st.write("Making predictions...")
    # Load the PyCaret model
    model = load_model('model')
    predictions = model.predict(keypoints_df)

    # Display the predictions
    st.write("Yoga form is ", predictions[0])

    ######### Sub-poses
        
    # Reload the CSV file
    #file_path = '/mnt/data/yolo-keypoints.csv'
    #data = pd.read_csv(file_path)

    # Create column names for data frame used for prediction
    columns = []
    for i in range(1, 31):
        columns.append(str(i) + '_' + "person")
    
        for key, value in KEYPOINT_DICT.items():
            columns.extend([str(i) + '_' + key + '_x', str(i) + '_' + key + '_y']) 
            
    data_columns = columns
    
    # Redefine the sub_poses and matching logic
    sub_poses = {
        'Sub-pose 1': (0, 3),
        'Sub-pose 2': (3, 8),
        'Sub-pose 3': (8, 12),
        'Sub-pose 4': (12, 22),
        'Sub-pose 5': (22, 30),
        'Sub-pose 6': (30, 35),
        'Sub-pose 7': (35, 37),
        'Sub-pose 8': (37, 40),
        'Sub-pose 09': (40, 46),
        'Sub-pose10': (46, 50),
        'Sub-pose 11': (50, 55),
        'Sub-pose 12': (55, 60)
    }
    
    def get_column_range_for_seconds(sub_pose_range):
        start_second, end_second = sub_pose_range
        columns = []
        for second in range(start_second, end_second + 1, 2):  # Step every 2 seconds
            column_prefix = f'{(second // 2) + 1}_'  # 2 seconds interval correspond to 1_, 2_, etc.
            matched_columns = [col for col in data_columns if col.startswith(column_prefix)]
            columns.extend(matched_columns)
        return columns
    
    # Create a dictionary mapping each sub-pose to its corresponding columns
    sub_pose_column_mapping = {sub_pose: get_column_range_for_seconds(time_range) for sub_pose, time_range in sub_poses.items()}
    
    # Display the result
    #st.write(sub_pose_column_mapping)

    subpose_number = 1

    # Create separate CSV files for each sub-pose including the 'class' column
    for sub_pose, columns in sub_pose_column_mapping.items():
        # Include the 'class' column along with the matched columns for each sub-pose

        # Dropping columns containing 'person', 'nose', 'eye', or 'ear' in their names
        #columns_to_drop = keypoints_df.filter(regex='person|nose|eye|ear').columns
        #keypoints_df = keypoints_df.drop(columns=columns_to_drop)
        #st.write(columns)

        # Define keywords to filter out
        keywords = ['person', 'nose', 'eye', 'ear']
        
        # Use list comprehension to filter out the items containing any of the keywords
        columns = [item for item in columns if not any(keyword in item for keyword in keywords)]
        
        sub_pose_data = keypoints_df[columns]

        #st.write(sub_pose_data)
        

        # Load the subpose model from the pickle file
        with open(f'Sub-pose_models/Sub-pose_{subpose_number}_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file) 
    
        st.write(f"{sub_pose} Form: ",loaded_model.predict(sub_pose_data)[0])

        subpose_number += 1
    
    
    
    
