# production_gradio_app.py

import gradio as gr
import pandas as pd
from gait_analysis_processor import (
    setup_database, add_user, get_user, hash_password,
    check_password, load_user_history, analyze_gait
)

# --- App Logic Functions ---
def signup(username, password):
    if not username or not password:
        raise gr.Error("Username and password cannot be empty.")
    hashed_pw = hash_password(password)
    if add_user(username, hashed_pw):
        return gr.Info("Signup successful! You can now log in.")
    else:
        raise gr.Error("Username already exists.")

def login(username, password, state):
    if not username or not password:
        raise gr.Error("Username and password cannot be empty.")
    user_record = get_user(username)
    if user_record and check_password(password, user_record[0]):
        state["logged_in"] = True
        state["username"] = username
        user_history = load_user_history(username)
        return (state, gr.update(visible=False), gr.update(visible=True), user_history)
    else:
        raise gr.Error("Invalid username or password.")

def logout(state):
    state["logged_in"] = False
    state["username"] = None
    # Reset all output components on logout
    return (state, gr.update(visible=True), gr.update(visible=False), None, "", {}, None, None, None, None)

def run_analysis_for_user(video_file, prosthetic_side, state, progress=gr.Progress(track_tqdm=True)):
    if not state["logged_in"]:
        raise gr.Error("You must be logged in to run an analysis.")
    if video_file is None:
        raise gr.Error("Please upload a video file.")
    
    # Robustly handle the video file path
    if hasattr(video_file, 'name'):
        video_path = video_file.name
    else:
        video_path = video_file
    
    def progress_callback(fraction):
        progress(fraction, desc=f"Analyzing... {int(fraction*100)}%")
    
    # Unpack all results from the backend
    video_out_path, summary, df, knee_fig, pelvic_fig = analyze_gait(
        state["username"], video_path, prosthetic_side, progress_callback
    )
    
    summary_text = (
        f"Video: {summary['video_name']}\n"
        f"Total Steps: {summary['total_steps']}\n"
        f"Avg. Stride Length: {summary['avg_stride_length_px']} (pixels)\n"
        f"Knee Asymmetry: {summary['asymmetry']}°\n"
        f"Pelvic Tilt Range (Drop): {summary['pelvic_tilt_range']}°"
    )
    
    user_history = load_user_history(state["username"])
    
    # Create a colored label for fall risk
    risk_color_map = {"Very Low": "green", "Low": "green", "Moderate": "orange", "High": "red"}
    fall_risk_label = gr.Label(
        value=summary['fall_risk'], 
        label="Fall Risk Assessment",
        color=risk_color_map.get(summary['fall_risk'], "grey")
    )

    return video_out_path, summary_text, fall_risk_label, df, knee_fig, pelvic_fig, user_history

# --- Gradio UI using Blocks ---
with gr.Blocks(theme=gr.themes.Soft(), title="NGPME") as app:
    gr.Markdown("# 🚶‍♂️ Next-Gen Prosthetic Mobility Enhancer (NGPME)")
    
    auth_state = gr.State({"logged_in": False, "username": None})

    with gr.Row(visible=True) as login_ui:
        with gr.Column(scale=1):
            gr.Markdown("## Login"); login_user = gr.Textbox(label="Username"); login_pass = gr.Textbox(label="Password", type="password"); login_btn = gr.Button("Login", variant="primary")
        with gr.Column(scale=1):
            gr.Markdown("## Signup"); signup_user = gr.Textbox(label="New Username"); signup_pass = gr.Textbox(label="New Password", type="password"); signup_btn = gr.Button("Sign Up")

    with gr.Row(visible=False) as main_app_ui:
        with gr.Column(scale=1):
            gr.Markdown("## Configuration"); video_input = gr.Video(label="Upload Video"); prosthetic_side_input = gr.Dropdown(["Left", "Right"], label="Prosthetic Side", value="Right"); analyze_button = gr.Button("Analyze Gait", variant="primary"); logout_btn = gr.Button("Logout")
        
        with gr.Column(scale=2):
            gr.Markdown("## Results")
            with gr.Tabs():
                with gr.TabItem("Summary & Video"):
                    fall_risk_output = gr.Label(label="Fall Risk Assessment")
                    with gr.Row():
                        video_output = gr.Video(label="Annotated Video")
                        summary_output = gr.Textbox(label="Analysis Summary", lines=5, interactive=False)
                    with gr.Accordion("Show Detailed Data", open=False):
                        detail_df_output = gr.DataFrame()
                
                with gr.TabItem("Performance Charts"):
                    gr.Markdown("Visualize your performance metrics over the course of the analysis.")
                    knee_plot_output = gr.Plot(label="Knee Angle Analysis")
                    pelvic_plot_output = gr.Plot(label="Pelvic Tilt Analysis")

                with gr.TabItem("My Session History"):
                    history_output = gr.DataFrame(label="Your Past Analyses")

    signup_btn.click(signup, inputs=[signup_user, signup_pass], outputs=None)
    login_btn.click(login, inputs=[login_user, login_pass, auth_state], outputs=[auth_state, login_ui, main_app_ui, history_output])
    logout_btn.click(logout, inputs=[auth_state], outputs=[auth_state, login_ui, main_app_ui, video_output, summary_output, fall_risk_output, detail_df_output, knee_plot_output, pelvic_plot_output, history_output])
    analyze_button.click(
        run_analysis_for_user,
        inputs=[video_input, prosthetic_side_input, auth_state],
        outputs=[video_output, summary_output, fall_risk_output, detail_df_output, knee_plot_output, pelvic_plot_output, history_output]
    )

if __name__ == "__main__":
    setup_database()
    app.launch()