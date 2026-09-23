
import requests
import config

TOKEN_URL = "https://accounts.spotify.com/api/token"
SEARCH_URL = "https://api.spotify.com/v1/search"

client_id = config.SPOTIFY_CLIENT_ID
client_secret = config.SPOTIFY_CLIENT_SECRET

print(f"Spotify client_id configured: {bool(client_id)}")
print(f"Spotify client_secret configured: {bool(client_secret)}")

if not client_id or not client_secret:
    raise ValueError(
        "Missing Spotify credentials. Check your .env and config.py"
    )

# Get Spotify access token
try:
    response = requests.post(
        TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        timeout=10,
    )

    print("Token HTTP status:", response.status_code)
    response.raise_for_status()

    token = response.json()["access_token"]
    print("Spotify authentication successful!\n")

except requests.RequestException as e:
    print("Spotify authentication failed:", e)
    raise SystemExit(1)

# Search Spotify for tracks
for label, query in [
    ("Upbeat electronic", "upbeat energetic electronic"),
    ("Focus music", "focus instrumental"),
    ("Relaxing music", "relaxing ambient"),
]:
    params = {
        "q": query,
        "type": "track",
        "limit": 5,
        "market": "IN",
    }

    headers = {
        "Authorization": f"Bearer {token}"
    }

    print(f"--- {label} ---")

    try:
        resp = requests.get(
            SEARCH_URL,
            params=params,
            headers=headers,
            timeout=10,
        )

        print("HTTP status:", resp.status_code)
        resp.raise_for_status()

        data = resp.json()
        tracks = data.get("tracks", {}).get("items", [])

        print("Results count:", len(tracks))

        for track in tracks[:3]:
            artists = ", ".join(
                artist["name"] for artist in track.get("artists", [])
            )

            print(
                "   ",
                track.get("name"),
                "-",
                artists,
                "| Spotify URL:",
                track.get("external_urls", {}).get("spotify"),
            )

    except requests.RequestException as e:
        print("REQUEST ERROR:", type(e).__name__, e)

    print()
