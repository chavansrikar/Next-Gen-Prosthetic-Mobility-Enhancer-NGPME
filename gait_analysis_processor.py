# gait_analysis_processor.py

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import os
import ffmpeg
import bcrypt
import plotly.express as px
import math

# --- (All helper functions up to analyze_gait are unchanged) ---
def hash_password(password): return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
def check_password(password, hashed): return bcrypt.checkpw(password.encode('utf-8'), hashed)
def add_user(username, password_hash, db_name="gait_analysis.db"):
    conn = sqlite3.connect(db_name); cursor = conn.cursor()
    try: cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash)); conn.commit(); return True
    except sqlite3.IntegrityError: return False
    finally: conn.close()
def get_user(username, db_name="gait_analysis.db"):
    conn = sqlite3.connect(db_name); cursor = conn.cursor(); cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,)); user_record = cursor.fetchone(); conn.close(); return user_record
def setup_database(db_name="gait_analysis.db"):
    conn = sqlite3.connect(db_name); cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, username TEXT, timestamp TEXT, video_name TEXT, total_steps INTEGER, avg_left_knee_angle REAL, avg_right_knee_angle REAL, asymmetry REAL, avg_pelvic_tilt REAL, pelvic_tilt_range REAL, avg_stride_length_px REAL, fall_risk TEXT, FOREIGN KEY (username) REFERENCES users (username))''')
    conn.commit(); conn.close()
def save_session_to_db(username, summary_data, db_name="gait_analysis.db"):
    conn = sqlite3.connect(db_name); cursor = conn.cursor()
    cursor.execute('''INSERT INTO sessions (username, timestamp, video_name, total_steps, avg_left_knee_angle, avg_right_knee_angle, asymmetry, avg_pelvic_tilt, pelvic_tilt_range, avg_stride_length_px, fall_risk) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (username, summary_data['timestamp'], summary_data['video_name'], summary_data['total_steps'], summary_data['avg_left_knee_angle'], summary_data['avg_right_knee_angle'], summary_data['asymmetry'], summary_data['avg_pelvic_tilt'], summary_data['pelvic_tilt_range'], summary_data['avg_stride_length_px'], summary_data['fall_risk']))
    conn.commit(); conn.close()
def load_user_history(username, db_name="gait_analysis.db"):
    conn = sqlite3.connect(db_name); history_df = pd.read_sql_query("SELECT * FROM sessions WHERE username = ? ORDER BY timestamp DESC", conn, params=(username,)); conn.close(); return history_df
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c); radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0]); angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle
def calculate_horizontal_angle(p1, p2): return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
def assess_fall_risk(asymmetry, pelvic_tilt_range, stride_lengths):
    score = 0
    if asymmetry > 8: score += 1
    if pelvic_tilt_range > 5: score += 1
    if len(stride_lengths) > 1 and np.std(stride_lengths) > 15: score += 1
    if score >= 2: return "Moderate"
    if score == 1: return "Low"
    return "Very Low"

def analyze_gait(username, video_path, prosthetic_side, progress_callback):
    setup_database(); mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(video_path)
    
    TARGET_WIDTH = 720
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = int(cap.get(cv2.CAP_PROP_FPS))
    if total_frames == 0: raise ValueError("Video file is invalid or empty.")
    if original_width > TARGET_WIDTH: output_width, output_height = TARGET_WIDTH, int(TARGET_WIDTH * (original_height / original_width))
    else: output_width, output_height = original_width, original_height
    
    temp_output_filename = 'temp_annotated_video.mp4'; fourcc = cv2.VideoWriter_fourcc(*'mp4v'); out = cv2.VideoWriter(temp_output_filename, fourcc, fps, (output_width, output_height))
    
    step_count = 0; gait_data = []; is_stepping = False; frame_count = 0; foot_down_pos = None; stride_lengths = []

    # --- OVERLAY COLOR CHANGE: Define connections for each leg ---
    L_LEG_CONNECTIONS = frozenset([(mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
                                   (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE)])
    R_LEG_CONNECTIONS = frozenset([(mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                                   (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)])
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1; progress_callback(frame_count / total_frames)
        if original_width > TARGET_WIDTH: frame = cv2.resize(frame, (output_width, output_height))
        
        image_for_processing = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); image_for_processing.flags.writeable = False; results = pose.process(image_for_processing); frame.flags.writeable = True
                    
        try:
            landmarks = results.pose_landmarks.landmark
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * output_width, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * output_height]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * output_width, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * output_height]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * output_width, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * output_height]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * output_width, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * output_height]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * output_width, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * output_height]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * output_width, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * output_height]
            
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle); right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            knee_angle_threshold = 160; active_knee_angle = left_knee_angle if prosthetic_side == 'Left' else right_knee_angle
            active_ankle_pos_x = left_ankle[0] if prosthetic_side == 'Left' else right_ankle[0]
            
            if active_knee_angle > knee_angle_threshold and not is_stepping:
                is_stepping = True; step_count += 1
                if foot_down_pos is not None:
                    stride = abs(active_ankle_pos_x - foot_down_pos); stride_lengths.append(stride)
                foot_down_pos = active_ankle_pos_x
            elif active_knee_angle < knee_angle_threshold and is_stepping: is_stepping = False
            pelvic_angle = calculate_horizontal_angle(left_hip, right_hip)
            gait_data.append({'frame': frame_count, 'left_knee_angle': left_knee_angle, 'right_knee_angle': right_knee_angle, 'pelvic_tilt': pelvic_angle})
        except:
            pass
        
        # --- OVERLAY COLOR CHANGE: Draw skeleton with different colors ---
        # 1. Draw the full skeleton with standard colors first
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=1)
        )

        # 2. Define the prosthetic color and connections
        prosthetic_color = (0, 255, 255) # Yellow
        prosthetic_connections = L_LEG_CONNECTIONS if prosthetic_side == "Left" else R_LEG_CONNECTIONS
        
        # 3. Draw over the prosthetic leg with the new color
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, prosthetic_connections,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=prosthetic_color, thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=prosthetic_color, thickness=2, circle_radius=2)
        )
        # --- END OVERLAY COLOR CHANGE ---

        out.write(frame)
        
    cap.release(); out.release(); pose.close()
    final_output_filename = 'annotated_video_final.mp4'
    try: (ffmpeg.input(temp_output_filename).output(final_output_filename, vcodec='libx264', pix_fmt='yuv4e0p').run(overwrite_output=True, quiet=True))
    except ffmpeg.Error as e: print("FFmpeg Error:", e.stderr.decode() if e.stderr else "Unknown FFmpeg error"); final_output_filename = temp_output_filename
    if os.path.exists(temp_output_filename) and final_output_filename != temp_output_filename: os.remove(temp_output_filename)
    if not gait_data: return None, "Analysis failed.", None, None, None
    df = pd.DataFrame(gait_data); smoothing_window = 5
    df['left_knee_smoothed'] = df['left_knee_angle'].rolling(window=smoothing_window, min_periods=1).mean()
    df['right_knee_smoothed'] = df['right_knee_angle'].rolling(window=smoothing_window, min_periods=1).mean()
    df['pelvic_tilt_smoothed'] = df['pelvic_tilt'].rolling(window=smoothing_window, min_periods=1).mean()
    knee_fig = px.line(df, x='frame', y=['left_knee_smoothed', 'right_knee_smoothed'], title="Knee Angle Over Time", labels={'value': 'Knee Angle (°)', 'frame': 'Frame Number'})
    pelvic_fig = px.line(df, x='frame', y='pelvic_tilt_smoothed', title="Pelvic Tilt Over Time", labels={'pelvic_tilt_smoothed': 'Pelvic Angle (°)'}); pelvic_fig.update_layout(showlegend=False)
    avg_left_knee = df['left_knee_smoothed'].mean(); avg_right_knee = df['right_knee_smoothed'].mean()
    asymmetry = abs(avg_left_knee - avg_right_knee); avg_pelvic_tilt = df['pelvic_tilt_smoothed'].mean()
    pelvic_tilt_range = df['pelvic_tilt_smoothed'].max() - df['pelvic_tilt_smoothed'].min()
    avg_stride_length_px = np.mean(stride_lengths) if stride_lengths else 0
    fall_risk = assess_fall_risk(asymmetry, pelvic_tilt_range, stride_lengths)
    summary_data = {'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'video_name': os.path.basename(video_path), 'total_steps': step_count, 'avg_left_knee_angle': round(avg_left_knee, 2), 'avg_right_knee_angle': round(avg_right_knee, 2), 'asymmetry': round(asymmetry, 2), 'avg_pelvic_tilt': round(avg_pelvic_tilt, 2), 'pelvic_tilt_range': round(pelvic_tilt_range, 2), 'avg_stride_length_px': round(avg_stride_length_px, 2), 'fall_risk': fall_risk}
    save_session_to_db(username, summary_data)
    return final_output_filename, summary_data, df, knee_fig, pelvic_fig