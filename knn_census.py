import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

# ---------------------------------------------------------------
# SUBCULTURE DEFINITIONS
# ---------------------------------------------------------------
subcultures = {
    0: "Inner-City Creative",
    1: "Aspirational Westie",
    2: "Cultural Enclave",
    3: "Surf-Urbanite",
    4: "Diaspora Professional",
    5: "Harbor Elite",
    6: "Vertical Cosmopolitan",
    7: "Semi-Rural Lifestyle Dweller",
}

# ---------------------------------------------------------------
# FAKE CENSUS PROFILE PER SUBCULTURE
# Each tuple: (median_age, weekly_income, pct_born_overseas,
#              pct_renting, pct_english_only, pct_university)
# Values reflect real ABS trends for each subculture type.
# ---------------------------------------------------------------
profiles = {
    #                          age    income  overseas  rent   english  uni
    0: dict(age=(30, 5),  income=(900,  120), overseas=(35, 10), rent=(72, 8),  english=(55, 10), uni=(60, 8)),   # Inner-City Creative
    1: dict(age=(37, 6),  income=(680,  100), overseas=(45, 12), rent=(38, 10), english=(52, 12), uni=(28, 8)),   # Aspirational Westie
    2: dict(age=(35, 5),  income=(580,   90), overseas=(72, 10), rent=(50, 10), english=(25, 12), uni=(22, 7)),   # Cultural Enclave
    3: dict(age=(42, 7),  income=(1150, 180), overseas=(18,  8), rent=(32,  8), english=(82,  8), uni=(52, 9)),   # Surf-Urbanite
    4: dict(age=(48, 6),  income=(1350, 200), overseas=(78,  8), rent=(28,  8), english=(20,  8), uni=(58, 8)),   # Diaspora Professional
    5: dict(age=(50, 6),  income=(3500, 500), overseas=(25,  8), rent=(18,  6), english=(72,  8), uni=(68, 8)),   # Harbor Elite
    6: dict(age=(31, 4),  income=(950,  130), overseas=(60, 10), rent=(65,  8), english=(18,  8), uni=(65, 8)),   # Vertical Cosmopolitan
    7: dict(age=(45, 7),  income=(1400, 250), overseas=(18,  8), rent=(15,  6), english=(78,  8), uni=(22, 7)),   # Semi-Rural Lifestyle Dweller
}

# ---------------------------------------------------------------
# SUBURB ANCHORS  (lon, lat, subculture_id, name)
# ---------------------------------------------------------------
suburb_anchors = [
    # Inner-City Creative
    (151.18, -33.89, 0, "Newtown"),
    (151.21, -33.88, 0, "Surry Hills"),
    (151.15, -33.91, 0, "Marrickville"),
    # Aspirational Westie
    (150.91, -33.77, 1, "Blacktown"),
    (150.84, -33.77, 1, "Rooty Hill"),
    (150.92, -33.93, 1, "Liverpool"),
    (151.00, -33.81, 1, "Parramatta"),
    (150.69, -33.75, 1, "Penrith"),
    (150.55, -33.70, 1, "Springwood"),
    (150.81, -34.07, 1, "Campbelltown"),
    (150.77, -33.76, 1, "St Marys"),
    (150.82, -33.77, 1, "Mount Druitt"),
    # Cultural Enclave
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
    # Surf-Urbanite
    (151.28, -33.80, 3, "Manly"),
    (151.15, -34.05, 3, "Cronulla"),
    (151.10, -33.70, 3, "Hornsby"),
    (151.30, -33.68, 3, "Mona Vale"),
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

# ---------------------------------------------------------------
# SUBURB ETHNICITY TAGS
# ---------------------------------------------------------------
ethnicity_tags = {
    "Newtown":          "Anglo/Creative",
    "Surry Hills":      "Anglo/Creative",
    "Marrickville":     "Greek/Vietnamese",
    "Blacktown":        "Filipino/Multicultural",
    "Rooty Hill":       "Filipino/Pacific Islander",
    "Liverpool":        "Lebanese/Iraqi/Indian",
    "Parramatta":       "Indian/Chinese",
    "Cabramatta":       "Vietnamese",
    "Fairfield":        "Vietnamese/Iraqi/Khmer",
    "Harris Park":      "Indian",
    "Lakemba":          "Lebanese/Bangladeshi",
    "Auburn":           "Turkish/Lebanese",
    "Bankstown":        "Lebanese/Multicultural",
    "Rockdale":         "Nepalese/Chinese",
    "Campsie":          "Chinese/Korean",
    "Strathfield":      "Korean/Chinese",
    "Hurstville":       "Chinese",
    "Eastwood":         "Chinese/Korean",
    "Wetherill Park":   "Lebanese/Iraqi",
    "Manly":            "Anglo-Australian",
    "Cronulla":         "Anglo-Australian",
    "Hornsby":          "Anglo-Australian",
    "Mona Vale":        "Anglo-Australian",
    "Penrith":          "Anglo-Australian",
    "Springwood":       "Anglo-Australian",
    "Campbelltown":     "Multicultural",
    "St Marys":         "Multicultural",
    "Mount Druitt":     "Pacific Islander/Aboriginal",
    "Stanhope Gardens": "Filipino/Indian",
    "The Ponds":         "Filipino/Chinese/Indian",
    "Rouse Hill":        "Filipino/Indian/Chinese",
    "Castle Hill":       "Chinese/Indian/Filipino",
    "Kellyville":       "Chinese/Indian",
    "Baulkham Hills":   "Chinese/Filipino",
    "Chatswood":        "Chinese/Korean",
    "Ryde":             "Chinese/Multicultural",
    "Epping":           "Chinese/Indian",
    "Vaucluse":         "Anglo-Australian/European",
    "Bellevue Hill":    "Anglo-Australian/Jewish",
    "Mosman":           "Anglo-Australian",
    "Double Bay":       "Anglo-Australian/European",
    "Rhodes":           "Chinese/Mandarin",
    "Zetland":          "Chinese/Indian",
    "Wolli Creek":      "Chinese/Korean",
    "Wentworth Point":  "Chinese/Mandarin",
    "Dural":            "Anglo-Australian",
    "Galston":          "Anglo-Australian",
    "Cobbitty":         "Anglo-Australian/Italian",
    "Kemps Creek":      "Anglo-Australian/Lebanese",
}

POINTS_PER_SUBURB = 30
GEO_SPREAD = 0.025

rows = []
for lon, lat, label, suburb in suburb_anchors:
    p = profiles[label]
    n = POINTS_PER_SUBURB
    for _ in range(n):
        rows.append({
            "suburb":           suburb,
            "subculture_id":    label,
            "subculture_name":  subcultures[label],
            "longitude":        round(np.random.normal(lon, GEO_SPREAD), 5),
            "latitude":         round(np.random.normal(lat, GEO_SPREAD), 5),
            "median_age":       round(np.random.normal(*p["age"])),
            "weekly_income":    round(np.random.normal(*p["income"])),
            "pct_born_overseas":round(np.clip(np.random.normal(*p["overseas"]), 0, 100), 1),
            "pct_renting":      round(np.clip(np.random.normal(*p["rent"]),     0, 100), 1),
            "pct_english_only": round(np.clip(np.random.normal(*p["english"]),  0, 100), 1),
            "pct_university":   round(np.clip(np.random.normal(*p["uni"]),      0, 100), 1),
        })

df = pd.DataFrame(rows)
df.to_csv("fake_census.csv", index=False)
print(f"Generated {len(df)} rows ({len(suburb_anchors)} suburbs x {POINTS_PER_SUBURB} points) -> fake_census.csv\n")

# ---------------------------------------------------------------
# TRAIN KNN ON ALL 7 FEATURES (not just location!)
# ---------------------------------------------------------------
FEATURES = ["longitude", "latitude", "median_age", "weekly_income",
            "pct_born_overseas", "pct_renting", "pct_english_only", "pct_university"]

X = df[FEATURES].values
y = df["subculture_id"].values

# Scale features so income doesn't dominate over percentages
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

acc = accuracy_score(y_test, knn.predict(X_test))
print(f"KNN Accuracy on held-out test set: {acc*100:.1f}%\n")
print(classification_report(y_test, knn.predict(X_test),
      target_names=[subcultures[i] for i in range(len(subcultures))]))

# ---------------------------------------------------------------
# PREDICT A MYSTERY PERSON — FULL DIAGNOSTIC REPORT
# ---------------------------------------------------------------
mystery = {
    "longitude":         151.05,   # <- somewhere near Burwood
    "latitude":         -33.87,
    "median_age":        33,
    "weekly_income":     620,
    "pct_born_overseas": 68,       # very high overseas-born
    "pct_renting":       55,
    "pct_english_only":  28,       # speaks another language at home
    "pct_university":    25,
}

mystery_df = pd.DataFrame([mystery])
mystery_scaled = scaler.transform(mystery_df[FEATURES])
pred = knn.predict(mystery_scaled)[0]

# Fit KNN on full dataset for richer neighbour lookup (train KNN used only for accuracy)
knn_full = KNeighborsClassifier(n_neighbors=5)
knn_full.fit(X_scaled, y)
distances, neighbor_idxs = knn_full.kneighbors(mystery_scaled, return_distance=True)
distances     = distances[0]
neighbor_idxs = neighbor_idxs[0]

W = 57
print("=" * W)
print("   MYSTERY PERSON — DIAGNOSTIC REPORT")
print("=" * W)
print(f"\nProfile: age={mystery['median_age']}, income=${mystery['weekly_income']}/wk, "
      f"{mystery['pct_born_overseas']}% born overseas, speaks other language at home")
print(f"\nKNN Prediction:  >>> {subcultures[pred]} <<<\n")

# ── Nearest Neighbours ──────────────────────────────────────────
print("── Nearest Neighbours " + "─" * (W - 22))
for rank, (idx, dist) in enumerate(zip(neighbor_idxs, distances), 1):
    suburb    = df.iloc[idx]["suburb"]
    ethnicity = ethnicity_tags.get(suburb, "Unknown")
    label     = f"{suburb} ({ethnicity})"
    print(f"  #{rank}  {label:<36}  dist: {dist:.3f}")

suburb_counts = Counter(df.iloc[i]["suburb"] for i in neighbor_idxs)
breakdown = ", ".join(f"{s}\u00d7{c}" for s, c in suburb_counts.most_common())
print(f"\n  Suburb breakdown: {breakdown}\n")

# ── Feature Contributions ───────────────────────────────────────
print("── Why This Prediction? Feature Contributions " + "─" * (W - 46))
neighbor_scaled = X_scaled[neighbor_idxs]                        # (5, 8)
per_feature_sq  = (mystery_scaled - neighbor_scaled) ** 2        # (5, 8)
mean_sq         = per_feature_sq.mean(axis=0)                    # (8,)
total           = mean_sq.sum()
pct             = mean_sq / total * 100 if total > 0 else np.zeros(len(FEATURES))

BAR_MAX = 20
order = np.argsort(pct)[::-1]
print(f"  {'Feature':<22}  {'':20}  {'%':>5}")
print(f"  {'─'*22}  {'─'*20}  {'─'*5}")
for i in order:
    bar = "\u2588" * int(round(pct[i] / 100 * BAR_MAX))
    print(f"  {FEATURES[i]:<22}  {bar:<20}  {pct[i]:>4.1f}%")
