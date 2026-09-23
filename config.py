
import os
from dotenv import load_dotenv

load_dotenv()

# EEG configuration
EEG_SOURCE = os.getenv("EEG_SOURCE", "simulation")
SIMULATION_MODE = "dataset_replay"

SAMPLE_RATE_HZ = 250
WINDOW_SECONDS = 5
WINDOW_SAMPLES = SAMPLE_RATE_HZ * WINDOW_SECONDS

BUFFER_SECONDS = 10
BUFFER_SAMPLES = SAMPLE_RATE_HZ * BUFFER_SECONDS

DATASET_PATH = "data/merged_eeg_datas_50_subjects.csv"
DATASET_SUBJECT_ID = None
DATASET_TASK_ID = None

ARDUINO_PORT = "COM5"
ARDUINO_BAUD_RATE = 115200

# Spotify configuration
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")

SPOTIFY_MARKET = "IN"
SPOTIFY_TRACK_LIMIT = 10

# Search terms based on predicted cognitive state
MOOD_TAGS = {
    "high": ["ambient", "calm", "chillout"],
    "moderate": ["instrumental", "focus", "lofi"],
    "low": ["upbeat", "energetic", "electronic"],
}

TRACKS_PER_RECOMMENDATION = 5
MUSIC_CACHE_TTL_SECONDS = 600

STATE_VOTE_WINDOW = 5
MIN_STATE_DWELL_SECONDS = 25

MUSIC_FRAGMENT_INTERVAL_SECONDS = 3
FEEDBACK_HALFLIFE_DAYS = 7