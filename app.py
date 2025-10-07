# app.py

import gradio as gr
import pandas as pd
import sqlite3 # --- DB ---
import os
from gait_analysis_processor import analyze_gait

# --- DB --- Function to load all historical data from the database
def load_history():
    if not os.path.exists("gait_analysis.db"):
        return pd.DataFrame() # Return empty dataframe if DB doesn't exist
    conn = sqlite3.connect("gait_analysis.db")
    # Read the entire 'sessions' table into a pandas DataFrame
    history_df = pd.read_sql_query("SELECT * FROM sessions ORDER BY timestamp DESC", conn)
    conn.close()
    return history_df

def video_analysis_interface(video_file, prosthetic_side):
    if video_file is None:
        return None, "Please upload a video file first.", None, None, load_history()

    video_path = video_file
    print("Starting gait analysis...")
    annotated_video, summary, data_df, data_csv = analyze_gait(video_path, prosthetic_side)
    print("Analysis complete.")
    
    # After analysis, return all results and refresh the history view
    return annotated_video, summary, data_df, data_csv, load_history()

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# Next-Gen Prosthetic Mobility Enhancer (NGPME)")
    
    with gr.Tabs():
        with gr.TabItem("Live Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 1. Configuration")
                    video_input = gr.Video(label="Upload Video for Analysis")
                    prosthetic_side_input = gr.Dropdown(["Left", "Right"], label="Prosthetic Side", value="Right")
                    analyze_button = gr.Button("Analyze Video", variant="primary")
                with gr.Column(scale=2):
                    gr.Markdown("## 2. Analysis Results")
                    video_output = gr.Video(label="Annotated Video Playback")
                    summary_output = gr.Textbox(label="Video Summary")
                    with gr.Accordion("Show Detailed Frame-by-Frame Data", open=False):
                        dataframe_output = gr.Dataframe(label="Gait Metrics")
                    file_output = gr.File(label="Download Analysis Data (CSV)")
        
        # --- DB --- New tab to display the history from the database
        with gr.TabItem("Session History"):
            gr.Markdown("## Past Analysis Sessions")
            history_df_output = gr.Dataframe(label="History")
            refresh_button = gr.Button("Refresh History")

    # When the app loads, and after an analysis, update the history
    app.load(load_history, None, history_df_output)
    refresh_button.click(load_history, None, history_df_output)
    
    analyze_button.click(
        fn=video_analysis_interface,
        inputs=[video_input, prosthetic_side_input],
        # The outputs list now includes the history dataframe
        outputs=[video_output, summary_output, dataframe_output, file_output, history_df_output]
    )

if __name__ == "__main__":
    app.launch()