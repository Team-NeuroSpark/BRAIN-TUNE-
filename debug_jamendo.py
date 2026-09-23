
import requests
import config

print(f"client_id = {config.JAMENDO_CLIENT_ID}\n")

for label, param_name, tag_value in [
    ("fuzzytags (OR)", "fuzzytags", "upbeat+energetic+electronic"),
    ("single tag: upbeat", "fuzzytags", "upbeat"),
    ("plain tags (AND)", "tags", "upbeat+energetic+electronic"),
]:
    params = {
        "client_id": config.JAMENDO_CLIENT_ID,
        "format": "json",
        "limit": 5,
        param_name: tag_value,
        "order": "popularity_total",
        "audioformat": "mp32",
    }
    print(f"--- {label} ---")
    print("URL params:", params)
    try:
        resp = requests.get("https://api.jamendo.com/v3.0/tracks/", params=params, timeout=8)
        print("HTTP status:", resp.status_code)
        data = resp.json()
        print("headers:", data.get("headers"))
        print("results_count:", len(data.get("results", [])))
        for t in data.get("results", [])[:3]:
            print("   ", t.get("name"), "-", t.get("artist_name"), "| audio:", bool(t.get("audio")))
    except Exception as e:
        print("EXCEPTION:", type(e).__name__, e)
    print()
