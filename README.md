# Next-Gen Prosthetic Mobility Enhancer (NGPME) 🚶‍♂️

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

NGPME is an advanced, web-based application that leverages computer vision to provide detailed gait analysis for prosthetic limb users. By simply uploading a video of themselves walking, users can receive objective, data-driven feedback on their mobility, helping them and their clinicians monitor progress and identify areas for improvement.



---
## ## Key Features

* 🔐 **Secure User Authentication**: Full signup and login system with bcrypt password hashing to ensure user data is private and secure.
* 👁️ **Computer Vision Gait Tracking**: Utilizes Google's MediaPipe Pose model to accurately detect 33 body landmarks without needing special markers or equipment.
* 📊 **Key Performance Metrics**: Automatically calculates critical gait parameters, including:
    * Total Step Count
    * Left & Right Knee Angles
    * Gait Asymmetry (comparing the motion of both legs)
* 🚀 **Performance Optimized**: Incorporates Region of Interest (ROI) Tracking and intelligent frame skipping to ensure fast analysis times on standard hardware.
* 📈 **Data Smoothing for Accuracy**: Raw angle data is processed with a Simple Moving Average (SMA) filter to reduce model jitter and provide more stable results.
* 🎨 **Interactive Data Visualization**: Generates an interactive Plotly chart showing knee angle flexion and extension over time.
* 🗄️ **Persistent User History**: All analysis sessions are automatically saved to a SQLite database, linked to the user's account for tracking progress.

---
## ## Tech Stack

* **Backend**: Python
* **UI Framework**: Gradio
* **Computer Vision**: OpenCV, MediaPipe
* **Data Handling**: Pandas, NumPy
* **Database**: SQLite
* **Plotting**: Plotly
* **Security**: bcrypt for password hashing
* **Video Processing**: FFmpeg

---
## ## Getting Started

Follow these steps to set up and run the project on your local machine.

### ### Prerequisites

* [Python 3.8+](https://www.python.org/downloads/)
* [Git](https://git-scm.com/downloads)
* [FFmpeg](https://ffmpeg.org/download.html) (Ensure it's added to your system's PATH)

### ### Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/chavansrikar/Next-Gen-Prosthetic-Mobility-Enhancer-NGPME-.git](https://github.com/chavansrikar/Next-Gen-Prosthetic-Mobility-Enhancer-NGPME-.git)
    cd Next-Gen-Prosthetic-Mobility-Enhancer-NGPME-
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv .venv
    .\.venv\Scripts\Activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python production_gradio_app.py
    ```
    Open your web browser and navigate to the local URL provided (usually `http://127.0.0.1:7860`).

---
## ## Our Mission ❤️

Technology should be a force for good, empowering everyone. NGPME is built on this principle. We are committed to developing an impactful, high-quality tool for disabled individuals and prosthetic users, providing them with valuable insights into their mobility **completely free of cost**. Our goal is to make advanced gait analysis accessible to all, helping users on their journey to improved movement and confidence.
