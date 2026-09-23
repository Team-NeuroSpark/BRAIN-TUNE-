# NeuroSpark

## Team Name
NeuroSpark  


## Team Members
- Faheemah M — Team Leader
- Ananya V Shahapur
- Harshitha N C

## Problem Statement
Most productivity and focus tools rely on static playlists or generic "focus music" that don't respond to how a person is actually doing cognitively in the moment — someone might be overloaded, understimulated, or in steady focus, and the music never adapts to reflect that.

## Solution Overview
NeuroSpark is a live dashboard that reads EEG signals, classifies cognitive workload (**low / moderate / high**) using a trained Random Forest model, and adapts the music being played based on that state in real time — calm/ambient when overloaded, upbeat when understimulated, and steady instrumental for sustained focus. The full loop is: **EEG signal → feature extraction → Random Forest classification → mood-tagged music recommendation (Jamendo API) → live Streamlit dashboard with like/skip feedback**, closing the gap between raw brain activity and an actual adaptive experience.

## PPT Link


## Live Demonstration Link
[View Presentation](https://drive.google.com/file/d/1wzkMxBb6dFUizY4VHZdbcbrVaxwZx3fd/view?usp=drivesdk)

## Deployed Link
https://neuro-spark.onrender.com

## Technology Stack
- **Frontend/Dashboard:** Streamlit
- **ML Model:** scikit-learn (Random Forest classifier)
- **Signal Processing:** NumPy, SciPy (band-power feature extraction)
- **Music API:** Jamendo API
- **Hardware:** BioAmp EXG Pill + Arduino (live EEG acquisition)
- **Data:** Pandas, joblib (model persistence)
- **Language:** Python

## Why
Most "focus music" apps just play a static playlist. NeuroSpark tries to close the loop: **read your brain state → determine what you need → adapt the music**, all in real time.

## Setup Instructions

```bash
pip install -r requirements.txt
```

Get a free Jamendo client ID from [Jamendo Developer Portal](https://devportal.jamendo.com), then:

```bash
cp .env.example .env
```

Edit `.env` and add your `JAMENDO_CLIENT_ID`.

Then run:

```bash
streamlit run streamlit_app.py
```

By default, NeuroSpark replays real EEG samples from the dataset (`SIMULATION_MODE = "dataset_replay"` in `config.py`), so no hardware is required. To use the actual Arduino board, switch `EEG_SOURCE` to `"arduino"` in `config.py`.

## Model

A **Random Forest** model trained on 8 band-power features is used for cognitive workload classification. Evaluation uses a **held-out subject split**, ensuring the model is not trained and tested on the same person.

The current held-out macro-F1 is around **0.53**. Cognitive workload classification from EEG is a challenging 3-class problem, so this represents realistic rather than inflated performance.

See `notebooks/01_eda_and_training.ipynb` for the complete EDA and training workflow.

## Project Structure

```text
neurospark/
├── streamlit_app.py          # Main dashboard
├── eeg_source.py             # EEG input abstraction
├── music_engine.py           # Jamendo recommendations + feedback logging
├── config.py                 # Configuration and settings
├── train.py                  # Model training script
├── debug_jamendo.py          # Jamendo API testing script
├── src/
│   ├── features.py           # Band-power feature extraction
│   └── dataset.py            # Training feature table generation
├── models/                   # Trained classifier + metadata
├── data/                     # EEG dataset
├── notebooks/                # EDA + training notebook
└── arduino/                  # Arduino firmware
```