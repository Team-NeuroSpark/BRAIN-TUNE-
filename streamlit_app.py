import os
import sys
import time
from collections import Counter, deque
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from fpdf import FPDF

# Make project src folder importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import config
from eeg_source import create_eeg_source
from features import extract_features, FEATURE_NAMES
import music_engine


st.set_page_config(page_title="NeuroSpark | Adaptive Music", layout="wide")

st.markdown("""
<style>
  .stApp {
    background: radial-gradient(circle at 12% 0%, #1c3150 0, #0b1220 42%, #070b13 100%);
    color: #eef4ff;
  }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111d31, #0a101c);
    border-right: 1px solid rgba(148, 163, 184, .18);
  }
  .block-container { padding-top: 2rem; max-width: 1380px; }
  h1, h2, h3 { color: #f3f7ff !important; }
  [data-testid="stMetric"] {
    background: rgba(24, 39, 63, .72);
    border: 1px solid rgba(125, 170, 220, .22);
    border-radius: 16px;
    padding: 14px 18px;
  }
  div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px;
    border-color: rgba(125, 170, 220, .24);
    background: rgba(17, 29, 48, .62);
  }
  .stButton button, .stDownloadButton button, .stLinkButton a {
    border-radius: 12px !important;
    font-weight: 650 !important;
  }
  .ns-hero {
    padding: 26px 30px; border-radius: 22px; margin: 4px 0 22px;
    background: linear-gradient(115deg, rgba(33, 86, 130, .72), rgba(91, 61, 145, .62));
    border: 1px solid rgba(180, 205, 255, .25);
    box-shadow: 0 18px 50px rgba(0,0,0,.22);
  }
  .ns-kicker { text-transform: uppercase; letter-spacing: .18em; color: #b6d8ff; font-size: .76rem; font-weight: 700; }
  .ns-hero h1 { margin: 7px 0 8px; font-size: 2.25rem; }
  .ns-hero p { margin: 0; color: #d4e3f7; font-size: 1.02rem; }
</style>
<div class="ns-hero">
  <div class="ns-kicker">NeuroSpark · EEG adaptive sound</div>
  <h1>Find your focus flow.</h1>
  <p>Live cognitive-state insights paired with instrumental, lyric-free music discovery.</p>
</div>
""", unsafe_allow_html=True)


MODEL_PATH = os.path.join(BASE_DIR, "models", "workload_classifier.joblib")


# -------------------- Session initialization --------------------
if "eeg_source" not in st.session_state:
    st.session_state.eeg_source = create_eeg_source()
    st.session_state.eeg_source.start()
source = st.session_state.eeg_source

if "model" not in st.session_state:
    st.session_state.model = joblib.load(MODEL_PATH)
model = st.session_state.model

defaults = {
    "shown_track_ids": set(),
    "current_music_state": None,
    "current_tracks": [],
    "liked_track_ids": set(),
    "state_votes": deque(maxlen=config.STATE_VOTE_WINDOW),
    "confirmed_state": None,
    "last_state_switch_ts": 0.0,
    "latest_raw_state": None,
    "latest_proba": {},
    "latest_features": {},
    "latest_window_stats": {},
    "session_start_ts": time.time(),
    "session_log": [],
    "tracks_played": [],
}
for key, value in defaults.items():
    if key not in st.session_state:
        # Create independent mutable objects for each session key.
        st.session_state[key] = value.copy() if isinstance(value, (set, list, dict)) else value
if "state_votes" not in st.session_state or not isinstance(st.session_state.state_votes, deque):
    st.session_state.state_votes = deque(maxlen=config.STATE_VOTE_WINDOW)


def _record_track(track, cognitive_state):
    """Add a recommended track to the session export log."""
    st.session_state.tracks_played.append({
        "ts": time.time(),
        "name": track.get("name", "Unknown track"),
        "artist": track.get("artist_name", "Unknown artist"),
        "cognitive_state": cognitive_state,
        "tag": track.get("tag", ""),
        "spotify_url": track.get("url", ""),
    })


# -------------------- Sidebar --------------------
@st.fragment(run_every=1)
def render_sidebar_metrics():
    st.metric("Total samples received", source.total_samples_received)
    st.metric(
        "Buffer fill",
        f"{source.buffered_samples}/{config.BUFFER_SAMPLES}",
    )
    if source.error:
        st.error(f"EEG source error: {source.error}")


with st.sidebar:
    render_sidebar_metrics()

st.caption("LIVE EEG • ADAPTIVE INSTRUMENTAL MUSIC")

# -------------------- EEG chart and prediction --------------------
@st.fragment(run_every=1)
def render_chart():
    window = source.get_latest_window(config.WINDOW_SAMPLES)

    if window is None:
        st.info(
            f"Buffering... {source.buffered_samples}/{config.WINDOW_SAMPLES} samples "
            f"(need {config.WINDOW_SECONDS}s of data before the first window is ready)."
        )
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Last {config.WINDOW_SECONDS}s of raw EEG")
        st.line_chart(window)

    with col2:
        st.subheader("Window stats")
        st.metric("Mean", f"{window.mean():.1f}")
        st.metric("Std dev", f"{window.std():.1f}")
        st.metric("Min / Max", f"{window.min():.0f} / {window.max():.0f}")

    feats = extract_features(window, fs=source.sample_rate)
    if feats is None:
        st.warning("Signal too flat/short this window: skipping prediction.")
        return

    x = pd.DataFrame(
        [[feats[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )
    raw_state = model.predict(x)[0]
    proba = dict(zip(model.classes_, model.predict_proba(x)[0]))
    st.session_state.latest_features = dict(feats)
    st.session_state.latest_window_stats = {
        "mean": float(window.mean()),
        "std": float(window.std()),
        "min": float(window.min()),
        "max": float(window.max()),
        "samples": int(len(window)),
    }

    st.session_state.state_votes.append(raw_state)
    st.session_state.latest_raw_state = raw_state
    st.session_state.latest_proba = proba
    # Keep a rolling history so simulation behavior can be diagnosed visibly.
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = deque(maxlen=60)
    st.session_state.prediction_history.append(str(raw_state))


# -------------------- Cognitive state and Spotify recommendations --------------------
@st.fragment(run_every=config.MUSIC_FRAGMENT_INTERVAL_SECONDS)
def render_state_and_music():
    votes = st.session_state.state_votes
    raw_state = st.session_state.latest_raw_state
    proba = st.session_state.latest_proba

    st.divider()
    col_a, col_b = st.columns([1, 2])

    if not votes:
        with col_a:
            st.info("Waiting for predictions...")
        return

    vote_counts = Counter(votes)
    majority_state, majority_count = vote_counts.most_common(1)[0]
    vote_confidence = majority_count / len(votes)

    now = time.time()
    dwell_elapsed = now - st.session_state.last_state_switch_ts
    can_switch = dwell_elapsed >= config.MIN_STATE_DWELL_SECONDS
    is_first_confirmation = st.session_state.confirmed_state is None

    should_switch = (
        majority_state != st.session_state.confirmed_state
        and (is_first_confirmation or can_switch)
    )

    if should_switch:
        st.session_state.confirmed_state = majority_state
        st.session_state.last_state_switch_ts = now

    cognitive_state = st.session_state.confirmed_state
    st.session_state.session_log.append({
        "ts": time.time(),
        "cognitive_state": cognitive_state,
    })

    with col_a:
        st.subheader("Cognitive state")
        st.metric("Driving music", str(cognitive_state).upper())
        st.caption(
            f"Instantaneous: {raw_state} · vote confidence: {vote_confidence:.0%}"
        )
        if proba:
            st.caption(
                " · ".join(
                    f"{key}: {value:.0%}"
                    for key, value in sorted(proba.items())
                )
            )
        with st.expander("Simulation / prediction diagnostics"):
            history = list(st.session_state.get("prediction_history", []))
            counts = Counter(history)
            st.write("Recent raw predictions (last 60):", dict(counts))
            if history:
                low_fraction = counts.get("low", 0) / len(history)
                if low_fraction >= 0.8:
                    st.warning(
                        f"Low dominates {low_fraction:.0%} of recent raw predictions. "
                        "This is a model/data diagnostic—not proof that the user's workload is low."
                    )
            st.write("EEG source:", getattr(config, "EEG_SOURCE", "unknown"))
            st.write("Source sample rate:", getattr(source, "sample_rate", "unknown"), "Hz")
            st.write("Configured window:", config.WINDOW_SECONDS, "seconds")
            st.write("Expected samples per window:", config.WINDOW_SAMPLES)
            st.write("Model classes:", [str(c) for c in getattr(model, "classes_", [])])
            st.write("Latest model probability output:", {
                str(k): round(float(v), 4) for k, v in proba.items()
            })
            st.write("Latest EEG window stats:", st.session_state.get("latest_window_stats", {}))
            with st.expander("Latest extracted features"):
                feature_values = st.session_state.get("latest_features", {})
                if feature_values:
                    st.dataframe(
                        pd.DataFrame([feature_values]).T.rename(columns={0: "value"}),
                        use_container_width=True,
                    )
                else:
                    st.info("Waiting for a valid EEG window.")
            st.caption(
                "If Low dominates, verify that simulation replays labeled examples across all classes, "
                "and that training and inference use the same sampling rate, preprocessing, feature order, "
                "and model pipeline. The app intentionally does not force or fabricate states."
            )
        if not can_switch and not is_first_confirmation:
            remaining = max(
                0,
                config.MIN_STATE_DWELL_SECONDS - dwell_elapsed,
            )
            st.caption(
                f"Playlist locked for {remaining:.0f}s more "
                "(prevents rapid swapping)"
            )

    with col_b:
        st.subheader("Adapted for this state")
        # Instrumental-focused search terms; avoid vocal/lyrical genres.
        instrumental_tags = {
            "low": ["upbeat instrumental", "light lo-fi beats", "cheerful instrumental"],
            "moderate": ["focus instrumental", "steady lo-fi beats", "soft jazz instrumental"],
            "high": ["calm ambient instrumental", "soft piano instrumental", "meditation music instrumental"],
        }
        state_key = str(cognitive_state).lower()
        tags = instrumental_tags.get(
            state_key,
            instrumental_tags["moderate"],
        )
        best_tag = music_engine.preferred_tag(cognitive_state, tags)

        state_changed = cognitive_state != st.session_state.current_music_state
        if state_changed:
            tracks = music_engine.recommend_tracks(
                cognitive_state,
                exclude_ids=st.session_state.shown_track_ids,
                preferred_tag=best_tag,
            )
            st.session_state.current_music_state = cognitive_state
            st.session_state.current_tracks = tracks

            for track in tracks:
                st.session_state.shown_track_ids.add(track["id"])

            if tracks:
                _record_track(tracks[0], cognitive_state)

            if len(st.session_state.shown_track_ids) > 100:
                st.session_state.shown_track_ids = set(
                    list(st.session_state.shown_track_ids)[-100:]
                )
        else:
            tracks = st.session_state.current_tracks

        if music_engine.last_error:
            st.warning(f"Spotify: {music_engine.last_error}")

        if not tracks:
            st.info(
                "No Spotify recommendations are available right now. "
                "Check your credentials/API access, then try again."
            )
            if st.button("Try Spotify again", key="spotify_retry"):
                st.session_state.current_music_state = None
                st.rerun(scope="fragment")
            return

        now_playing = tracks[0]

        with st.container(border=True):
            img_col, info_col = st.columns([1, 3])
            with img_col:
                if now_playing.get("image"):
                    st.image(now_playing["image"], width=110)

            with info_col:
                st.caption("🎧 SPOTIFY RECOMMENDATION")
                st.markdown(f"### {now_playing.get('name', 'Unknown track')}")
                st.markdown(f"*{now_playing.get('artist_name', 'Unknown artist')}*")
                duration_ms = now_playing.get("duration_ms", 0)
                if duration_ms:
                    seconds = duration_ms // 1000
                    duration_text = f"{seconds // 60}:{seconds % 60:02d}"
                    st.caption(
                        f"{now_playing.get('tag', '').title()} · {duration_text}"
                    )
                else:
                    st.caption(now_playing.get("tag", "").title())

                spotify_url = now_playing.get("url")
                if spotify_url:
                    track_id = spotify_url.rstrip("/").split("/")[-1].split("?")[0]
                    if track_id:
                        import streamlit.components.v1 as components
                        components.iframe(
                            f"https://open.spotify.com/embed/track/{track_id}",
                            height=152,
                            scrolling=False,
                        )

            already_liked = (
                now_playing["id"] in st.session_state.liked_track_ids
            )
            fb_cols = st.columns(2)
            like_label = "❤️ Liked" if already_liked else "👍 Like"

            if fb_cols[0].button(
                like_label,
                key=f"up_{now_playing['id']}",
                use_container_width=True,
                disabled=already_liked,
            ):
                music_engine.log_feedback(
                    now_playing["id"],
                    cognitive_state,
                    now_playing.get("tag", ""),
                    True,
                )
                st.session_state.liked_track_ids.add(now_playing["id"])
                st.toast(
                    f"Liked — will favor '{now_playing.get('tag', '')}' "
                    f"more for {cognitive_state} state."
                )
                st.rerun(scope="fragment")

            if fb_cols[1].button(
                "👎 Skip",
                key=f"down_{now_playing['id']}",
                use_container_width=True,
            ):
                music_engine.log_feedback(
                    now_playing["id"],
                    cognitive_state,
                    now_playing.get("tag", ""),
                    False,
                )
                replacement = music_engine.swap_track(
                    cognitive_state,
                    now_playing.get("tag", ""),
                    st.session_state.shown_track_ids,
                )

                if replacement:
                    st.session_state.shown_track_ids.add(replacement["id"])
                    st.session_state.current_tracks[0] = replacement
                    _record_track(replacement, cognitive_state)
                    st.toast(
                        f"Skipped — next recommendation: "
                        f"{replacement.get('name', 'track')}"
                    )
                else:
                    st.warning(
                        "No new Spotify track was found. Try again later."
                    )
                st.rerun(scope="fragment")


# -------------------- Session export --------------------

@st.fragment(run_every=config.MUSIC_FRAGMENT_INTERVAL_SECONDS)
def render_session_export():
    st.divider()
    st.subheader("Session summary")

    if not st.session_state.session_log:
        return

    state_counts = Counter(
        row["cognitive_state"] for row in st.session_state.session_log
    )
    total_windows = sum(state_counts.values())
    fb_stats = music_engine.get_session_feedback_stats(
        st.session_state.session_start_ts
    )

    col1, col2 = st.columns(2)
    with col1:
        st.caption("State distribution (this session)")
        for state, count in state_counts.items():
            st.write(f"{state}: {count / total_windows:.0%}")

    with col2:
        st.caption("Feedback")
        if fb_stats["total"]:
            ratio = fb_stats["like_ratio"]
            ratio_text = f"{ratio:.0%}" if ratio is not None else "N/A"
            st.write(
                f"👍 {fb_stats['likes']} · 👎 {fb_stats['dislikes']} "
                f"({ratio_text} like ratio)"
            )
        else:
            st.write("No likes/skips yet")

    csv_df = pd.DataFrame(st.session_state.tracks_played)
    csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download CSV",
        csv_bytes,
        file_name="neurospark_session.csv",
        mime="text/csv",
    )


render_chart()
render_state_and_music()
render_session_export()