
import json
import os
import time
import base64

import requests
import config

TOKEN_URL = "https://accounts.spotify.com/api/token"
SEARCH_URL = "https://api.spotify.com/v1/search"

FEEDBACK_LOG_PATH = os.path.join("data", "feedback_log.json")

last_error = None
_token = None
_token_expires_at = 0


def get_access_token():
    """Get and cache a Spotify app access token."""
    global _token, _token_expires_at

    if _token and time.time() < _token_expires_at - 60:
        return _token

    client_id = config.SPOTIFY_CLIENT_ID
    client_secret = config.SPOTIFY_CLIENT_SECRET

    if not client_id or not client_secret:
        raise ValueError(
            "Spotify credentials are missing. "
            "Check SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env."
        )

    credentials = f"{client_id}:{client_secret}"
    encoded = base64.b64encode(credentials.encode()).decode()

    response = requests.post(
        TOKEN_URL,
        headers={
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"grant_type": "client_credentials"},
        timeout=15,
    )
    response.raise_for_status()

    data = response.json()
    _token = data["access_token"]
    _token_expires_at = time.time() + data.get("expires_in", 3600)

    return _token


def _search_tracks(query, limit=10):
    """Search Spotify and return normalized track objects."""
    token = get_access_token()

    response = requests.get(
        SEARCH_URL,
        headers={"Authorization": f"Bearer {token}"},
        params={
            "q": query,
            "type": "track",
            "limit": min(max(int(limit), 1), 10),
            "market": config.SPOTIFY_MARKET,
        },
        timeout=15,
    )
    response.raise_for_status()

    data = response.json()
    items = data.get("tracks", {}).get("items", [])

    tracks = []

    for item in items:
        if not item:
            continue

        artists = ", ".join(
            artist.get("name", "Unknown artist")
            for artist in item.get("artists", [])
        )

        album_images = item.get("album", {}).get("images", [])
        image_url = album_images[0]["url"] if album_images else ""

        external_urls = item.get("external_urls", {})
        spotify_url = external_urls.get("spotify", "")

        if not spotify_url:
            continue

        tracks.append({
            "id": item.get("id", spotify_url),
            "name": item.get("name", "Unknown track"),
            "artist_name": artists or "Unknown artist",
            "url": spotify_url,
            "image": image_url,
            "tag": query,
            "duration_ms": item.get("duration_ms", 0),
            "preview_url": item.get("preview_url"),
        })

    return tracks


def recommend_tracks(
    cognitive_state,
    exclude_ids=None,
    limit=None,
    preferred_tag=None,
):
    """Recommend Spotify tracks for the current cognitive state."""
    global last_error

    limit = limit or config.TRACKS_PER_RECOMMENDATION
    exclude_ids = exclude_ids or set()

    tags = config.MOOD_TAGS.get(
        cognitive_state,
        config.MOOD_TAGS["moderate"],
    )

    search_tags = [preferred_tag] if preferred_tag else tags

    try:
        tracks = []

        for tag in search_tags:
            results = _search_tracks(
                tag,
                limit=config.SPOTIFY_TRACK_LIMIT,
            )

            for track in results:
                if track["id"] not in exclude_ids:
                    track["tag"] = tag
                    tracks.append(track)

                if len(tracks) >= limit:
                    break

            if len(tracks) >= limit:
                break

        # If preferred tag didn't return enough, try the state's other tags.
        if len(tracks) < limit:
            for tag in tags:
                results = _search_tracks(
                    tag,
                    limit=config.SPOTIFY_TRACK_LIMIT,
                )

                for track in results:
                    if (
                        track["id"] not in exclude_ids
                        and track["id"] not in {
                            t["id"] for t in tracks
                        }
                    ):
                        track["tag"] = tag
                        tracks.append(track)

                    if len(tracks) >= limit:
                        break

                if len(tracks) >= limit:
                    break

        if not tracks:
            last_error = "Spotify returned no matching tracks."
            return []

        last_error = None
        return tracks[:limit]

    except (requests.RequestException, ValueError, KeyError) as error:
        last_error = f"{type(error).__name__}: {error}"
        return []


def _load_feedback_log():
    if not os.path.exists(FEEDBACK_LOG_PATH):
        return []

    try:
        with open(FEEDBACK_LOG_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return []


def get_session_feedback_stats(since_ts):
    log = _load_feedback_log()

    session_log = [
        row for row in log
        if row.get("ts", 0) >= since_ts
    ]

    likes = sum(1 for row in session_log if row.get("liked"))
    dislikes = len(session_log) - likes
    total = len(session_log)

    return {
        "likes": likes,
        "dislikes": dislikes,
        "total": total,
        "like_ratio": likes / total if total else None,
    }


def log_feedback(track_id, cognitive_state, tag, liked):
    os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH), exist_ok=True)

    log = _load_feedback_log()

    log.append({
        "track_id": track_id,
        "cognitive_state": cognitive_state,
        "tag": tag,
        "liked": bool(liked),
        "ts": time.time(),
    })

    with open(FEEDBACK_LOG_PATH, "w", encoding="utf-8") as file:
        json.dump(log, file, indent=2)


def preferred_tag(cognitive_state, default_tags):
    log = _load_feedback_log()
    now = time.time()

    halflife_seconds = config.FEEDBACK_HALFLIFE_DAYS * 86400
    scores = {}

    for tag in default_tags:
        rows = [
            row for row in log
            if row.get("cognitive_state") == cognitive_state
            and row.get("tag") == tag
        ]

        if len(rows) < 3:
            continue

        weighted_sum = 0.0
        weight_total = 0.0

        for row in rows:
            age = now - row.get("ts", now)

            weight = (
                0.5 ** (age / halflife_seconds)
                if halflife_seconds > 0
                else 1.0
            )

            weighted_sum += weight * (
                1.0 if row.get("liked") else 0.0
            )
            weight_total += weight

        if weight_total > 0:
            scores[tag] = weighted_sum / weight_total

    if not scores:
        return default_tags[0]

    return max(scores, key=scores.get)


def swap_track(cognitive_state, tag, exclude_ids):
    global last_error

    tracks = recommend_tracks(
        cognitive_state,
        exclude_ids=exclude_ids,
        limit=1,
        preferred_tag=tag,
    )

    if tracks:
        return tracks[0]

    return None


def get_feedback_summary():
    log = _load_feedback_log()
    counts = {}

    for row in log:
        state = row.get("cognitive_state", "unknown")

        counts.setdefault(
            state,
            {
                "cognitive_state": state,
                "likes": 0,
                "dislikes": 0,
            },
        )

        if row.get("liked"):
            counts[state]["likes"] += 1
        else:
            counts[state]["dislikes"] += 1

    return list(counts.values())