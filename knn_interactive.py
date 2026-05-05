import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ---------------------------------------------------------------
# SUBCULTURE DEFINITIONS
# ---------------------------------------------------------------
subcultures = {
    0: "Inner-City Creative",
    1: "Aspirational Westie",
    2: "Cultural Enclave",
    3: "Surf-Urbanite",
}

profiles = {
    #                          age    income  overseas  rent   english  uni
    0: dict(age=(30, 5),  income=(900,  120), overseas=(35, 10), rent=(72, 8),  english=(55, 10), uni=(60, 8)),
    1: dict(age=(37, 6),  income=(680,  100), overseas=(45, 12), rent=(38, 10), english=(52, 12), uni=(28, 8)),
    2: dict(age=(35, 5),  income=(580,   90), overseas=(72, 10), rent=(50, 10), english=(25, 12), uni=(22, 7)),
    3: dict(age=(42, 7),  income=(1150, 180), overseas=(18,  8), rent=(32,  8), english=(82,  8), uni=(52, 9)),
}

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
    (151.28, -33.80, 3, "Manly"),
    (151.15, -34.05, 3, "Cronulla"),
    (151.10, -33.70, 3, "Hornsby"),
    (151.30, -33.68, 3, "Mona Vale"),
    (150.69, -33.75, 1, "Penrith"),
    (150.55, -33.70, 1, "Springwood"),
    (150.81, -34.07, 1, "Campbelltown"),
    (150.77, -33.76, 1, "St Marys"),
    (150.82, -33.77, 1, "Mount Druitt"),
]

ethnicity_tags = {
    "Newtown":        "Anglo/Creative",
    "Surry Hills":    "Anglo/Creative",
    "Marrickville":   "Greek/Vietnamese",
    "Blacktown":      "Multicultural",
    "Rooty Hill":     "Pacific Islander",
    "Liverpool":      "Arabic/Indian",
    "Parramatta":     "Indian/Chinese",
    "Cabramatta":     "Vietnamese",
    "Fairfield":      "Vietnamese/Khmer",
    "Harris Park":    "Indian",
    "Lakemba":        "Lebanese/Bangladeshi",
    "Auburn":         "Turkish/Lebanese",
    "Campsie":        "Chinese/Korean",
    "Strathfield":    "Chinese/Korean",
    "Hurstville":     "Chinese",
    "Eastwood":       "Chinese/Korean",
    "Wetherill Park": "Lebanese/Iraqi",
    "Manly":          "Anglo-Australian",
    "Cronulla":       "Anglo-Australian",
    "Hornsby":        "Anglo-Australian",
    "Mona Vale":      "Anglo-Australian",
    "Penrith":        "Anglo-Australian",
    "Springwood":     "Anglo-Australian",
    "Campbelltown":   "Multicultural",
    "St Marys":       "Multicultural",
    "Mount Druitt":   "Pacific Islander/Aboriginal",
}

# ---------------------------------------------------------------
# BUILD DATASET + TRAIN MODEL (silently)
# ---------------------------------------------------------------
FEATURES = ["longitude", "latitude", "median_age", "weekly_income",
            "pct_born_overseas", "pct_renting", "pct_english_only", "pct_university"]

rows = []
for lon, lat, label, suburb in suburb_anchors:
    p = profiles[label]
    for _ in range(30):
        rows.append({
            "suburb":            suburb,
            "subculture_id":     label,
            "longitude":         round(np.random.normal(lon, 0.025), 5),
            "latitude":          round(np.random.normal(lat, 0.025), 5),
            "median_age":        round(np.random.normal(*p["age"])),
            "weekly_income":     round(np.random.normal(*p["income"])),
            "pct_born_overseas": round(np.clip(np.random.normal(*p["overseas"]), 0, 100), 1),
            "pct_renting":       round(np.clip(np.random.normal(*p["rent"]),     0, 100), 1),
            "pct_english_only":  round(np.clip(np.random.normal(*p["english"]),  0, 100), 1),
            "pct_university":    round(np.clip(np.random.normal(*p["uni"]),      0, 100), 1),
        })

df = pd.DataFrame(rows)
X_scaled = StandardScaler().fit_transform(df[FEATURES].values)
scaler    = StandardScaler().fit(df[FEATURES].values)
X_scaled  = scaler.transform(df[FEATURES].values)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, df["subculture_id"].values)

# ---------------------------------------------------------------
# INPUT HELPERS
# ---------------------------------------------------------------
def ask_number(prompt, lo, hi):
    while True:
        try:
            val = float(input(prompt))
            if lo <= val <= hi:
                return val
            print(f"  Please enter a value between {lo} and {hi}.")
        except ValueError:
            print("  Please enter a number.")

def ask_yes_no(prompt):
    while True:
        ans = input(prompt).strip().lower()
        if ans in ("yes", "y"):
            return True
        if ans in ("no", "n"):
            return False
        print("  Please answer yes or no.")

def find_suburb(name):
    name_lower = name.lower().strip()
    for lon, lat, _, suburb in suburb_anchors:
        if suburb.lower() == name_lower:
            return lon, lat, suburb
    for lon, lat, _, suburb in suburb_anchors:
        if name_lower in suburb.lower():
            return lon, lat, suburb
    return None

# ---------------------------------------------------------------
# QUESTIONS
# Proxy mapping rationale:
#   Personal yes/no answers stand in for suburb-level census %s.
#   Values are set to the midpoint of each subculture's profile
#   for that feature, representing a "strong" signal either way.
# ---------------------------------------------------------------
W = 57
print("=" * W)
print("  Sydney Subculture Predictor — tell us about yourself")
print("=" * W)
print()

# Q1 — location
raw_suburb = input("[1/7] What suburb do you live in (or nearest known suburb)?\n      > ").strip()
match = find_suburb(raw_suburb)
if match:
    lon, lat, matched = match
    print(f"      Matched: {matched}\n")
else:
    lon, lat, matched = 151.21, -33.87, raw_suburb + " (unrecognised — using Sydney CBD)"
    print(f"      '{raw_suburb}' not in model, defaulting to Sydney CBD coordinates.\n")

# Q2 — age
age = int(ask_number("[2/7] How old are you?\n      > ", 10, 100))
print()

# Q3 — income
income = int(ask_number("[3/7] Roughly what is your weekly take-home income ($)?\n      > ", 0, 10000))
print()

# Q4 — born overseas  →  pct_born_overseas proxy
born_os = ask_yes_no("[4/7] Were you born overseas? (yes / no)\n      > ")
pct_overseas = 72 if born_os else 18
print()

# Q5 — language at home  →  pct_english_only proxy
other_lang = ask_yes_no("[5/7] Do you mainly speak a language other than English at home? (yes / no)\n      > ")
pct_english = 22 if other_lang else 80
print()

# Q6 — renting  →  pct_renting proxy
renting = ask_yes_no("[6/7] Do you currently rent your home? (yes / no)\n      > ")
pct_rent = 62 if renting else 28
print()

# Q7 — university  →  pct_university proxy
uni = ask_yes_no("[7/7] Do you have a university degree? (yes / no)\n      > ")
pct_uni = 55 if uni else 20
print()

# ---------------------------------------------------------------
# ASSEMBLE PROFILE + PREDICT
# ---------------------------------------------------------------
profile = {
    "longitude":         lon,
    "latitude":          lat,
    "median_age":        age,
    "weekly_income":     income,
    "pct_born_overseas": pct_overseas,
    "pct_renting":       pct_rent,
    "pct_english_only":  pct_english,
    "pct_university":    pct_uni,
}

profile_scaled = scaler.transform(np.array([[profile[f] for f in FEATURES]]))
pred           = knn.predict(profile_scaled)[0]
distances, neighbor_idxs = knn.kneighbors(profile_scaled, return_distance=True)
distances     = distances[0]
neighbor_idxs = neighbor_idxs[0]

# ---------------------------------------------------------------
# DIAGNOSTIC REPORT
# ---------------------------------------------------------------
print("=" * W)
print("  YOUR SYDNEY SUBCULTURE — DIAGNOSTIC REPORT")
print("=" * W)

summary = (
    f"age {age}, ${income}/wk, "
    + ("born overseas" if born_os else "born in Australia") + ", "
    + ("other language at home" if other_lang else "English at home") + ", "
    + ("renting" if renting else "owner-occupier") + ", "
    + ("uni degree" if uni else "no uni degree")
)
print(f"\nYour profile:  {summary}")
print(f"\nKNN Prediction:  >>> {subcultures[pred]} <<<\n")
print("  (Note: your personal answers are used as proxies for suburb-level")
print("   census statistics, so this is an approximation.)\n")

# -- Nearest Neighbours --
print("-- Nearest Neighbours " + "-" * (W - 22))
for rank, (idx, dist) in enumerate(zip(neighbor_idxs, distances), 1):
    suburb    = df.iloc[idx]["suburb"]
    ethnicity = ethnicity_tags.get(suburb, "Unknown")
    label     = f"{suburb} ({ethnicity})"
    print(f"  #{rank}  {label:<36}  dist: {dist:.3f}")

suburb_counts = Counter(df.iloc[i]["suburb"] for i in neighbor_idxs)
breakdown = ", ".join(f"{s}x{c}" for s, c in suburb_counts.most_common())
print(f"\n  Suburb breakdown: {breakdown}\n")

# -- Feature Contributions --
print("-- Why This Prediction? Feature Contributions " + "-" * (W - 46))
neighbor_scaled = X_scaled[neighbor_idxs]
per_feature_sq  = (profile_scaled - neighbor_scaled) ** 2
mean_sq         = per_feature_sq.mean(axis=0)
total           = mean_sq.sum()
pct             = mean_sq / total * 100 if total > 0 else np.zeros(len(FEATURES))

order = np.argsort(pct)[::-1]
print(f"  {'Feature':<22}  {'':20}  {'%':>5}")
print(f"  {'-'*22}  {'-'*20}  {'-'*5}")
for i in order:
    bar = "#" * int(round(pct[i] / 100 * 20))
    print(f"  {FEATURES[i]:<22}  {bar:<20}  {pct[i]:>4.1f}%")
