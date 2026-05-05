"""
generate_map_data.py
Exports the 780 synthetic residents as a JavaScript data file for
the interactive KNN map canvas on how-it-works.html.
Uses the same seed and parameters as knn_census.py.
"""

import numpy as np
import json

np.random.seed(42)

suburb_anchors = [
    (151.18, -33.89, 0, "Newtown"),
    (151.21, -33.88, 0, "Surry Hills"),
    (151.15, -33.91, 0, "Marrickville"),
    (150.91, -33.77, 1, "Blacktown"),
    (150.84, -33.77, 1, "Rooty Hill"),
    (150.92, -33.93, 1, "Liverpool"),
    (151.00, -33.81, 1, "Parramatta"),
    (150.94, -33.89, 2, "Cabramatta"),
    (150.96, -33.87, 2, "Fairfield"),
    (151.01, -33.82, 2, "Harris Park"),
    (151.07, -33.92, 2, "Lakemba"),
    (151.03, -33.85, 2, "Auburn"),
    (151.10, -33.91, 2, "Campsie"),
    (151.09, -33.88, 2, "Strathfield"),
    (151.10, -33.97, 2, "Hurstville"),
    (151.08, -33.79, 2, "Eastwood"),
    (150.90, -33.84, 2, "Wetherill Park"),
    (151.03, -33.92, 2, "Bankstown"),
    (151.14, -33.96, 2, "Rockdale"),
    (151.28, -33.80, 3, "Manly"),
    (151.15, -34.05, 3, "Cronulla"),
    (151.10, -33.70, 3, "Hornsby"),
    (151.30, -33.68, 3, "Mona Vale"),
    (150.69, -33.75, 1, "Penrith"),
    (150.55, -33.70, 1, "Springwood"),
    (150.81, -34.07, 1, "Campbelltown"),
    (150.77, -33.76, 1, "St Marys"),
    (150.82, -33.77, 1, "Mount Druitt"),
    # Diaspora Professional
    (150.87, -33.72, 4, "Stanhope Gardens"),
    (150.90, -33.70, 4, "The Ponds"),
    (150.91, -33.68, 4, "Rouse Hill"),
    (151.00, -33.73, 4, "Castle Hill"),
    (150.97, -33.68, 4, "Kellyville"),
    (150.98, -33.76, 4, "Baulkham Hills"),
    (151.18, -33.80, 4, "Chatswood"),
    (151.10, -33.82, 4, "Ryde"),
    (151.08, -33.77, 4, "Epping"),
    # Harbor Elite
    (151.28, -33.86, 5, "Vaucluse"),
    (151.26, -33.88, 5, "Bellevue Hill"),
    (151.24, -33.83, 5, "Mosman"),
    (151.25, -33.88, 5, "Double Bay"),
    # Vertical Cosmopolitan
    (151.09, -33.83, 6, "Rhodes"),
    (151.21, -33.91, 6, "Zetland"),
    (151.15, -33.95, 6, "Wolli Creek"),
    (151.07, -33.83, 6, "Wentworth Point"),
    # Semi-Rural Lifestyle Dweller
    (151.03, -33.69, 7, "Dural"),
    (151.02, -33.67, 7, "Galston"),
    (150.70, -34.00, 7, "Cobbitty"),
    (150.72, -33.88, 7, "Kemps Creek"),
]

rows = []
for lon, lat, label, suburb in suburb_anchors:
    for _ in range(30):
        rows.append({
            "lon": round(float(np.random.normal(lon, 0.025)), 4),
            "lat": round(float(np.random.normal(lat, 0.025)), 4),
            "s":   label,
            "sub": suburb,
        })

out = "const RESIDENTS = " + json.dumps(rows, separators=(",", ":")) + ";\n"

with open("static/residents_data.js", "w", encoding="utf-8") as f:
    f.write(out)

print(f"Written {len(rows)} residents → static/residents_data.js ({len(out)} bytes)")
