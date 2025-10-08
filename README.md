---
title: Next-Gen Prosthetic Mobility Enhancer
emoji: 🚶‍♂️
sdk: gradio
app_file: production_gradio_app.py
---

# Next-Gen Prosthetic Mobility Enhancer (NGPME) 🚶‍♂️

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Live Demo 🚀

You can try the live application hosted on Hugging Face Spaces:

**[➡️ Click here to open the Live Demo](https://huggingface.co/spaces/srikarrr/Next-Gen-Prosthetic-Mobility-Enhancer)**

---

NGPME is an advanced, web-based application that leverages computer vision to provide detailed gait analysis for prosthetic limb users. By simply uploading a video of themselves walking, users can receive objective, data-driven feedback on their mobility, helping them and their clinicians monitor progress and identify areas for improvement.

---
## Key Features

* 🔐 **Secure User Authentication**: Full signup and login system with bcrypt password hashing to ensure user data is private and secure.
* 👁️ **High-Accuracy Pose Estimation**: Utilizes Google's most powerful **MediaPipe Pose (Heavy)** model to provide the highest possible accuracy for joint detection.
* 🎨 **Customizable Overlay**: Generates a clean, thin-line skeletal overlay that highlights the user-selected prosthetic leg in a distinct color for easy visual analysis.
* 📊 **Comprehensive Gait Metrics**: Automatically calculates critical gait parameters, including:
    * Total Step Count & Average Stride Length (in pixels)
    * Left & Right Knee Angles
    * Gait Asymmetry
    * Pelvic Tilt & Pelvic Drop (Range)
* 🧠 **Fall Risk Assessment**: A rule-based model analyzes gait metrics to provide a "Very Low," "Low," or "Moderate" fall risk assessment for each session.
* 📈 **Data Smoothing & Visualization**: Raw numerical data is smoothed with a Simple Moving Average (SMA) filter to provide stable metrics, which are then displayed in interactive **Plotly charts**.
* 🗄️ **Persistent User History**: All analysis sessions are automatically saved to a SQLite database, linked to the user's account for tracking progress over time.

---
## Tech Stack

* **Backend**: Python
* **UI Framework**: Gradio
* **Computer Vision**: OpenCV, MediaPipe
* **Data Handling**: Pandas, NumPy
* **Database**: SQLite
* **Plotting**: Plotly
* **Security**: bcrypt for password hashing
* **Video Processing**: FFmpeg

---
## Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

* [Python 3.8+](https://www.python.org/downloads/)
* [Git](https://git-scm.com/downloads)
* [FFmpeg](https://ffmpeg.org/download.html) (Ensure it's added to your system's PATH)

### Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/chavansrikar/Next-Gen-Prosthetic-Mobility-Enhancer-NGPME.git](https://github.com/chavansrikar/Next-Gen-Prosthetic-Mobility-Enhancer-NGPME.git)
    cd Next-Gen-Prosthetic-Mobility-Enhancer-NGPME
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
## Upcoming Features

* **Live Webcam Analysis**: Implement a real-time analysis mode using a webcam feed for immediate feedback.
* **Real-World Unit Conversion**: Add a camera calibration feature to convert pixel-based measurements (like stride length) into real-world units (meters/centimeters).
* **PDF Report Generation**: Create a "Download Report" button that generates a professional PDF summarizing the analysis results.
* **Clinician Dashboard**: A future goal to create a separate interface for medical professionals to monitor the progress of multiple users who have granted them access.

---
## Our Mission ❤️

Technology should be a force for good, empowering everyone. NGPME is built on this principle. We are committed to developing an impactful, high-quality tool for disabled individuals and prosthetic users, providing them with valuable insights into their mobility **completely free of cost**. Our goal is to make advanced gait analysis accessible to all, helping users on their journey to improved movement and confidence.
