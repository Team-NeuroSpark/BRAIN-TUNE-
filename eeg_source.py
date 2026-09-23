from abc import ABC, abstractmethod
import threading
import time
import json
import random
from collections import deque

import numpy as np

import config


class _RingBuffer:
    def __init__(self, maxlen):
        self._buf = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self.total_samples_received = 0

    def push(self, value):
        with self._lock:
            self._buf.append(value)
            self.total_samples_received += 1

    def push_many(self, values):
        with self._lock:
            self._buf.extend(values)
            self.total_samples_received += len(values)

    def get_latest(self, n):
        with self._lock:
            if len(self._buf) < n:
                return None
            return np.array(list(self._buf)[-n:], dtype=np.float64)

    def __len__(self):
        with self._lock:
            return len(self._buf)

class EEGSource(ABC):
    sample_rate = config.SAMPLE_RATE_HZ

    def __init__(self):
        self._buffer = _RingBuffer(maxlen=config.BUFFER_SAMPLES)
        self._thread = None
        self._stop_flag = threading.Event()
        self.is_running = False
        self.error = None  
    @abstractmethod
    def _run(self):
        raise NotImplementedError

    def start(self):
        if self.is_running:
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.is_running = True

    def stop(self):
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.is_running = False

    def get_latest_window(self, n_samples=None):
        n_samples = n_samples or config.WINDOW_SAMPLES
        return self._buffer.get_latest(n_samples)

    @property
    def total_samples_received(self):
        return self._buffer.total_samples_received

    @property
    def buffered_samples(self):
        return len(self._buffer)

class SimulationEEGSource(EEGSource):

    def __init__(self, mode=None, dataset_path=None,
                 subject_id=None, task_id=None):
        super().__init__()
        self.mode = mode or config.SIMULATION_MODE
        self.dataset_path = dataset_path or config.DATASET_PATH
        self.subject_id = subject_id or config.DATASET_SUBJECT_ID
        self.task_id = task_id or config.DATASET_TASK_ID
        self._replay_samples = None  

        self.cognitive_state = "moderate"

    def _load_replay_samples(self):
        import pandas as pd

        df = pd.read_csv(self.dataset_path)

        if self.subject_id is not None:
            df = df[df["student_id"] == self.subject_id]
        if self.task_id is not None:
            df = df[df["task_id"] == self.task_id]

        if df.empty:
            raise ValueError(
                "No rows matched subject_id/task_id filters in dataset replay."
            )

        all_samples = []
        for raw in df["neuro_raw_eeg"]:
            arr = json.loads(raw) if isinstance(raw, str) else raw
            all_samples.extend(arr)

        if not all_samples:
            raise ValueError("Selected dataset rows had no raw_eeg samples.")

        return all_samples
    def _synthetic_sample(self, t):
        weights = {
            "relaxed":  {"alpha": 1.0, "beta": 0.3},
            "moderate": {"alpha": 0.6, "beta": 0.6},
            "focused":  {"alpha": 0.3, "beta": 1.0},
        }[self.cognitive_state]

        alpha = weights["alpha"] * np.sin(2 * np.pi * 10 * t)  
        beta = weights["beta"] * np.sin(2 * np.pi * 20 * t)     
        theta = 0.2 * np.sin(2 * np.pi * 6 * t)                
        mains_noise = 0.05 * np.sin(2 * np.pi * 50 * t)         
        broadband_noise = random.gauss(0, 0.15)

        value = 512 + 50 * (alpha + beta + theta + mains_noise + broadband_noise)
        return value

    def _run(self):
        period = 1.0 / self.sample_rate
        t0 = time.perf_counter()
        sample_index = 0

        if self.mode == "dataset_replay":
            self._replay_samples = self._load_replay_samples()
            replay_len = len(self._replay_samples)

        next_tick = time.perf_counter()
        while not self._stop_flag.is_set():
            if self.mode == "dataset_replay":
                value = self._replay_samples[sample_index % replay_len]
            else:
                t = sample_index / self.sample_rate
                value = self._synthetic_sample(t)

            self._buffer.push(value)
            sample_index += 1

            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
class ArduinoEEGSource(EEGSource):

    def __init__(self, port=None, baud_rate=None):
        super().__init__()
        self.port = port or config.ARDUINO_PORT
        self.baud_rate = baud_rate or config.ARDUINO_BAUD_RATE
        self._serial = None

    def _run(self):
        import serial 

        try:
            self._serial = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)  
        except Exception as e:
            self.error = f"Could not open serial port {self.port}: {e}"
            return

        while not self._stop_flag.is_set():
            try:
                line = self._serial.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                value = float(line)
                self._buffer.push(value)
            except ValueError:
                continue
            except Exception:
                time.sleep(0.05)
                continue

        if self._serial is not None:
            self._serial.close()

def create_eeg_source(source_type=None, **kwargs) -> EEGSource:
    source_type = source_type or config.EEG_SOURCE

    if source_type == "simulation":
        return SimulationEEGSource(**kwargs)
    elif source_type == "arduino":
        return ArduinoEEGSource(**kwargs)
    else:
        raise ValueError(
            f"Unknown EEG_SOURCE '{source_type}'. Use 'simulation' or 'arduino'."
        )
