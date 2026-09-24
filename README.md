# 🧠 BRAIN TUNE — EEG-Based Adaptive Music System

**Team NeuroSpark**

BRAIN TUNE is an EEG-based adaptive music system that analyzes brain signals to estimate cognitive workload and recommends music through the **Spotify Web API**.

The system supports both real-time EEG input and simulated EEG data, allowing users to explore workload estimation and adaptive music recommendations.

---
## Members
Faheemah M
Harshitha N C
Ananya V Shahapur 
----
## ✨ Features

* **EEG Signal Acquisition:** Supports EEG data from connected hardware and simulated signals.
* **Real-Time Processing:** Processes EEG signals in windows to extract relevant features.
* **Cognitive Workload Classification:** Uses a trained machine-learning model to classify workload levels.
* **Adaptive Music Recommendations:** Uses Spotify to search for tracks based on the estimated workload.
* **Interactive Dashboard:** Streamlit interface for viewing EEG signals, workload predictions, and music recommendations.
* **Feedback Logging:** Records user feedback to support future evaluation and improvements.

---

## 🧠 System Workflow

1. Acquire EEG data from hardware or use simulated data.
2. Preprocess the signal and extract relevant features.
3. Pass the features to the trained workload classification model.
4. Estimate the user's cognitive workload.
5. Use the workload result to guide music selection.
6. Search Spotify and display recommended tracks in the dashboard.

---

## 🛠️ Technology Stack

| Component            | Technology                               |
| -------------------- | ---------------------------------------- |
| Programming Language | Python                                   |
| Dashboard            | Streamlit                                |
| Machine Learning     | Scikit-learn                             |
| Signal Processing    | NumPy, SciPy                             |
| Data Processing      | Pandas                                   |
| Model Storage        | Joblib                                   |
| Music Integration    | Spotify Web API                          |
| EEG Hardware         | Arduino-compatible EEG acquisition setup |

---

## 📁 Project Structure

```text
BRAIN-TUNE/
├── arduino/
│   └── arduino_eeg_reader.ino
├── data/
│   ├── feedback_log.json
│   └── merged_eeg_datas_50_subjects.csv
├── models/
│   └── workload_classifier.joblib
├── notebooks/
│   └── 01_eda_and_training.ipynb
├── src/
│   ├── dataset.py
│   └── features.py
├── config.py
├── eeg_source.py
├── music_engine.py
├── streamlit_app.py
├── train.py
├── test_eeg_source.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Team-NeuroSpark/BRAIN-TUNE.git
cd BRAIN-TUNE
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt:

```cmd
venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Spotify API credentials

1. Open the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard).
2. Create or select an application.
3. Obtain the Spotify Client ID and Client Secret.
4. Create a `.env` file in the project root.
5. Add your credentials:

```env
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
```

**Keep your `.env` file private. Do not commit API credentials or secrets to GitHub.**

If sharing the project, provide a `.env.example` file containing only placeholder values.

---

## ▶️ Running the Application

From the project root, run:

```bash
streamlit run streamlit_app.py
```

Streamlit will provide a local URL to open the dashboard in your browser.

---

## 📡 EEG Input Modes

BRAIN TUNE is designed to support:

* **Real-time mode:** Receives EEG data from the configured hardware and serial connection.
* **Simulation mode:** Uses simulated EEG data for testing and demonstration when hardware is unavailable.

Check `config.py` to configure the EEG source, sampling rate, window duration, and hardware connection settings.

---

## 🎵 Spotify Integration

The music engine uses the Spotify Web API to search for tracks and provide Spotify links based on the system's workload classification.

Spotify integration requires valid API credentials configured through environment variables.

Music recommendations are intended for demonstration and research purposes. They are not medical advice or a substitute for professional assessment.

---

## 🧪 Model Training

The project includes a training script and notebook for dataset exploration and model development.

To run the training script:

```bash
python train.py
```

The trained model is stored under the `models/` directory when training completes successfully.

---
🔗 Project Links

###  Live Demo

**[Open NeuroSpark — BRAIN TUNE](https://brain-tune-neurospark-p7e1.onrender.com/)**

The deployed application demonstrates the NeuroSpark dashboard and its EEG-based adaptive music workflow.

### Video Demonstration

**[Watch the Project Demo](https://drive.google.com/file/d/1J0JbXwy3bliS2QfDTVFWKo8QAPGh5IB4/view?usp=sharing)**

The demonstration video covers the hardware setup, EEG acquisition, real-time dashboard, workload classification, and Spotify-based adaptive music recommendation.

###  Presentation

**[View Project Presentation](https://docs.google.com/presentation/d/1FFgX9_heGfrnNmV3dwQtXlpJy-XA7KKa/edit?usp=sharing&ouid=115081753363421060141&rtpof=true&sd=true)**


## 🧰 Troubleshooting

* **Spotify authentication errors:** Check that the Client ID and Client Secret are correct and loaded from `.env`.
* **No tracks returned:** Check your internet connection and Spotify API access.
* **EEG input not detected:** Verify the configured serial port, baud rate, wiring, and hardware connection.
* **Simulation not working:** Check the EEG source and simulation settings in `config.py`.
* **Missing Python packages:** Activate the virtual environment and run `pip install -r requirements.txt`.

---

## 🔐 Security Notes

* Never upload `.env` or expose Spotify credentials.
* Keep private keys, tokens, and personal data out of the repository.
* Use `.env.example` for documenting required environment variables.

---

## 👥 Team

**Team NeuroSpark**

Project: **BRAIN TUNE — EEG-Based Adaptive Music System**

---

## 📄 Disclaimer

BRAIN TUNE is a research and demonstration project. Cognitive workload estimates and music recommendations are experimental and should not be interpreted as clinical diagnoses or medical advice.
