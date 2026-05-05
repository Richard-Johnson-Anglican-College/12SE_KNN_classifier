import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# Subculture groups
# ---------------------------------------------------------------
subcultures = {
    0: "Inner-City Creative",
    1: "Aspirational Westie",
    2: "Cultural Enclave",
    3: "Surf-Urbanite",
}

# ---------------------------------------------------------------
# Suburb anchors — (longitude, latitude, subculture_id, suburb, ethnicity)
# ---------------------------------------------------------------
suburb_anchors = [
    # Inner-City Creative Class
    (151.18, -33.89, 0, "Newtown",      "Anglo-Australian / LGBTQ+"),
    (151.21, -33.88, 0, "Surry Hills",  "Anglo-Australian / Euro"),
    (151.15, -33.91, 0, "Marrickville", "Greek / Vietnamese"),

    # Aspirational West
    (150.91, -33.77, 1, "Blacktown",    "Filipino"),
    (150.84, -33.77, 1, "Rooty Hill",   "Filipino"),
    (150.92, -33.93, 1, "Liverpool",    "Lebanese / Arabic"),
    (151.00, -33.81, 1, "Parramatta",   "Indian / Pakistani"),

    # Cultural Enclaves
    (150.94, -33.89, 2, "Cabramatta",   "Vietnamese"),
    (150.96, -33.87, 2, "Fairfield",    "Assyrian / Iraqi"),
    (151.01, -33.82, 2, "Harris Park",  "Indian"),

    # Coastal & Leafy North
    (151.28, -33.80, 3, "Manly",          "Anglo-Australian"),
    (151.15, -34.05, 3, "Cronulla",       "Anglo-Australian"),
    (151.10, -33.70, 3, "Hornsby",        "Anglo-Australian / Chinese"),
    (151.30, -33.68, 3, "Mona Vale",      "Anglo-Australian"),

    # Outer West & South
    (150.69, -33.75, 1, "Penrith",        "Anglo-Australian"),
    (150.31, -33.71, 1, "Blue Mountains", "Anglo-Australian / Alternative"),
    (150.81, -34.07, 1, "Campbelltown",   "Anglo-Australian / Pacific Islander"),
    (150.77, -33.76, 1, "St Marys",       "Anglo-Australian / Pacific Islander"),
    (150.82, -33.77, 1, "Mount Druitt",   "Anglo-Australian / Indigenous / Pacific Islander"),

    # Inner-West & South Cultural Enclaves
    (151.07, -33.92, 2, "Lakemba",        "Lebanese / Bangladeshi / Muslim"),
    (151.03, -33.85, 2, "Auburn",         "Turkish / Lebanese / South Asian"),
    (151.10, -33.91, 2, "Campsie",        "Chinese / Korean / Greek"),
    (151.09, -33.88, 2, "Strathfield",    "Korean / Chinese"),
    (151.10, -33.97, 2, "Hurstville",     "Chinese"),
    (151.08, -33.79, 2, "Eastwood",       "Chinese / Korean"),
    (150.90, -33.84, 2, "Wetherill Park", "Lebanese / South Asian / Iraqi"),
]

# ---------------------------------------------------------------
# Generate synthetic "residents" jittered around each suburb
# ---------------------------------------------------------------
np.random.seed(42)
POINTS_PER_SUBURB = 25
SPREAD = 0.025

rows = []
for lon, lat, label, suburb, ethnicity in suburb_anchors:
    jitter_lons = np.random.normal(lon, SPREAD, POINTS_PER_SUBURB)
    jitter_lats = np.random.normal(lat, SPREAD, POINTS_PER_SUBURB)
    for jlon, jlat in zip(jitter_lons, jitter_lats):
        rows.append({
            "longitude":       round(jlon, 5),
            "latitude":        round(jlat, 5),
            "nearest_suburb":  suburb,
            "ethnicity":       ethnicity,
            "subculture_id":   label,
            "subculture_name": subcultures[label],
        })

df = pd.DataFrame(rows)
df.to_csv("sydney_subcultures.csv", index=False)

print(f"Saved {len(df)} rows to sydney_subcultures.csv\n")
print(df[["nearest_suburb", "ethnicity", "subculture_name"]].drop_duplicates().to_string(index=False))
