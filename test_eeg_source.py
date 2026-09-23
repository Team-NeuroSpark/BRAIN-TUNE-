import time
import config
from eeg_source import create_eeg_source

def main():
    print(f"EEG_SOURCE = {config.EEG_SOURCE}")
    source = create_eeg_source()
    source.start()

    try:
        for i in range(8):
            time.sleep(1)
            window = source.get_latest_window(config.WINDOW_SAMPLES)
            print(
                f"[t={i+1}s] total_received={source.total_samples_received:5d} "
                f"buffered={source.buffered_samples:5d} "
                f"window_ready={'YES' if window is not None else 'no'}"
                + (f"  sample_mean={window.mean():.1f}" if window is not None else "")
            )
    finally:
        source.stop()
        print("Stopped cleanly.")

if __name__ == "__main__":
    main()
