# gait_analysis_processor.py

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import os # <--- THIS LINE IS THE FIX

# --- DB --- Function to setup the database and create tables if they don't exist
def setup_database(db_name="gait_analysis.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # Create a table for analysis sessions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        video_name TEXT,
        total_steps INTEGER,
        avg_left_knee_angle REAL,
        avg_right_knee_angle REAL,
        asymmetry REAL
    )''')
    conn.commit()
    conn.close()

# --- DB --- Function to save the results of a session to the database
def save_session_to_db(summary_data, db_name="gait_analysis.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO sessions (timestamp, video_name, total_steps, avg_left_knee_angle, avg_right_knee_angle, asymmetry)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        summary_data['timestamp'],
        summary_data['video_name'],
        summary_data['total_steps'],
        summary_data['avg_left_knee_angle'],
        summary_data['avg_right_knee_angle'],
        summary_data['asymmetry']
    ))
    conn.commit()
    conn.close()

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def analyze_gait(video_path, prosthetic_side):
    # --- DB --- Make sure the database is ready before we start
    setup_database()
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(video_path)
    FRAME_SKIP = 3 
    TARGET_WIDTH = 1280
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if original_width > TARGET_WIDTH:
        aspect_ratio = original_height / original_width
        output_width = TARGET_WIDTH
        output_height = int(output_width * aspect_ratio)
    else:
        output_width = original_width
        output_height = original_height
    output_filename = 'annotated_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (output_width, output_height))
    roi_box = None; roi_padding = 50; failed_detections = 0; ROI_RESET_THRESHOLD = 15
    step_count = 0; gait_data = []; is_stepping = False; frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        if original_width > TARGET_WIDTH: frame = cv2.resize(frame, (output_width, output_height))
        if roi_box: x, y, w, h = roi_box; frame_for_processing = frame[y:y+h, x:x+w]
        else: frame_for_processing = frame
        image_for_processing = cv2.cvtColor(frame_for_processing, cv2.COLOR_BGR2RGB)
        image_for_processing.flags.writeable = False
        results = pose.process(image_for_processing)
        frame.flags.writeable = True
        try:
            landmarks = results.pose_landmarks.landmark
            failed_detections = 0
            h_roi, w_roi, _ = frame_for_processing.shape
            offset_x, offset_y = (roi_box[0], roi_box[1]) if roi_box else (0, 0)
            for lm in landmarks:
                lm.x = (lm.x * w_roi + offset_x) / output_width
                lm.y = (lm.y * h_roi + offset_y) / output_height
            if frame_count % FRAME_SKIP == 0:
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]; left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]; left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]; right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]; right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle); right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                knee_angle_threshold = 150; active_knee_angle = left_knee_angle if prosthetic_side == 'Left' else right_knee_angle
                if active_knee_angle > knee_angle_threshold and not is_stepping: is_stepping = True
                elif active_knee_angle < knee_angle_threshold and is_stepping: step_count += 1; is_stepping = False
                gait_data.append({'frame': frame_count, 'left_knee_angle': left_knee_angle, 'right_knee_angle': right_knee_angle, 'steps': step_count})
                cv2.putText(frame, f'Left Knee: {int(left_knee_angle)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA); cv2.putText(frame, f'Right Knee: {int(right_knee_angle)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA); cv2.putText(frame, f'Steps: {step_count}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            all_x = [lm.x * output_width for lm in landmarks]; all_y = [lm.y * output_height for lm in landmarks]
            min_x, max_x = int(min(all_x)), int(max(all_x)); min_y, max_y = int(min(all_y)), int(max(all_y))
            x = max(0, min_x - roi_padding); y = max(0, min_y - roi_padding); w = min(output_width - x, (max_x - min_x) + 2 * roi_padding); h = min(output_height - y, (max_y - min_y) + 2 * roi_padding)
            roi_box = (x, y, w, h)
        except:
            failed_detections += 1
            if failed_detections > ROI_RESET_THRESHOLD: roi_box = None
            pass
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        out.write(frame)

    cap.release(); out.release(); pose.close()
    df = pd.DataFrame(gait_data)
    csv_filename = 'gait_analysis_data.csv'; df.to_csv(csv_filename, index=False)

    if not df.empty:
        avg_left_knee = df['left_knee_angle'].mean()
        avg_right_knee = df['right_knee_angle'].mean()
        asymmetry = abs(avg_left_knee - avg_right_knee)
        summary_data_for_db = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'video_name': os.path.basename(video_path),
            'total_steps': step_count,
            'avg_left_knee_angle': round(avg_left_knee, 2),
            'avg_right_knee_angle': round(avg_right_knee, 2),
            'asymmetry': round(asymmetry, 2)
        }
        save_session_to_db(summary_data_for_db)
        summary_text = (f"--- Analysis Complete & Saved to DB ---\n"
                        f"Total Steps: {step_count}\n"
                        f"Avg Left Knee Angle: {avg_left_knee:.2f}\n"
                        f"Avg Right Knee Angle: {avg_right_knee:.2f}\n"
                        f"Asymmetry: {asymmetry:.2f}")
    else:
        summary_text = "Analysis could not be completed."

    return output_filename, summary_text, df, csv_filename