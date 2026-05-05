from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

app = Flask(__name__)

# ── Subculture metadata ───────────────────────────────────────────
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

PRED_CLASS = {
    "Inner-City Creative": "creative",
    "Aspirational Westie": "westie",
    "Cultural Enclave":    "enclave",
    "Surf-Urbanite":       "surf",
    "Diaspora Professional": "diaspora",
    "Harbor Elite":              "harbor",
    "Vertical Cosmopolitan":     "vertical",
    "Semi-Rural Lifestyle Dweller": "semirural",
}

PRED_DESCRIPTION = {
    "Inner-City Creative": "Renters in inner-city areas drawn to creative, academic and hospitality work. Typically young, university-educated, and diverse — with English as the dominant but not exclusive home language.",
    "Aspirational Westie": "Multicultural families and individuals in the outer-western suburbs. A mix of renters and owner-occupiers across a broad income range, united more by location and aspiration than by wealth.",
    "Cultural Enclave":    "Overseas-born residents living in close-knit communities where a language other than English is spoken at home. Household incomes vary widely; the defining trait is cultural and linguistic identity.",
    "Surf-Urbanite":       "Established owner-occupiers in coastal and upper-northern suburbs. Predominantly Anglo-Australian with English spoken exclusively at home — typically middle-aged or older, with above-average but variable household incomes.",
    "Diaspora Professional": "Overseas-born homeowners, largely university-educated, concentrated in the Hills District and north-western corridor. Non-English home language is common; incomes tend toward the higher end, reflecting strong professional employment.",
    "Harbor Elite": "Owner-occupiers in prestige harbourside suburbs of the Eastern Suburbs and Lower North Shore. Predominantly Anglo-Australian or European heritage, high English-only rates, and among the highest household incomes in Sydney.",
    "Vertical Cosmopolitan": "Renters in high-density apartments along rail corridors. Predominantly first-generation migrants from East Asia and South Asia, with non-English home language the norm. Typically young and university-educated.",
    "Semi-Rural Lifestyle Dweller": "Owner-occupiers on large blocks at the urban fringe, drawn by space and lifestyle. Predominantly Anglo-Australian or European heritage, English-only at home, with lower university rates. Household incomes are typically above average.",
}

# ── Census profiles (mean, std) per subculture ───────────────────
profiles = {
    0: dict(age=(30, 5),  income=(900,  120), overseas=(35, 10), rent=(72, 8),  english=(55, 10), uni=(60, 8)),
    1: dict(age=(37, 6),  income=(680,  100), overseas=(45, 12), rent=(38, 10), english=(52, 12), uni=(28, 8)),
    2: dict(age=(35, 5),  income=(580,   90), overseas=(72, 10), rent=(50, 10), english=(25, 12), uni=(22, 7)),
    3: dict(age=(42, 7),  income=(1150, 180), overseas=(18,  8), rent=(32,  8), english=(82,  8), uni=(52, 9)),
    4: dict(age=(48, 6),  income=(1350, 200), overseas=(78,  8), rent=(28,  8), english=(20,  8), uni=(58, 8)),   # Diaspora Professional
    5: dict(age=(50, 6),  income=(1800, 200), overseas=(25,  8), rent=(18,  6), english=(72,  8), uni=(68, 8)),   # Harbor Elite
    6: dict(age=(31, 4),  income=(950,  130), overseas=(60, 10), rent=(65,  8), english=(18,  8), uni=(65, 8)),   # Vertical Cosmopolitan
    7: dict(age=(45, 7),  income=(1400, 250), overseas=(18,  8), rent=(15,  6), english=(78,  8), uni=(22, 7)),   # Semi-Rural Lifestyle Dweller
}

# ── Suburb anchors (lon, lat, subculture_id, name) ───────────────
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
    # Diaspora Professional — Hills District & north-west corridor
    (150.87, -33.72, 4, "Stanhope Gardens"),
    (150.90, -33.70, 4, "The Ponds"),
    (150.91, -33.68, 4, "Rouse Hill"),
    (151.00, -33.73, 4, "Castle Hill"),
    (150.97, -33.68, 4, "Kellyville"),
    (150.98, -33.76, 4, "Baulkham Hills"),
    (151.18, -33.80, 4, "Chatswood"),
    (151.10, -33.82, 4, "Ryde"),
    (151.08, -33.77, 4, "Epping"),
    # Harbor Elite — Eastern Suburbs harbourside
    (151.28, -33.86, 5, "Vaucluse"),
    (151.26, -33.88, 5, "Bellevue Hill"),
    (151.24, -33.83, 5, "Mosman"),
    (151.25, -33.88, 5, "Double Bay"),
    # Vertical Cosmopolitan — high-density rail corridors
    (151.09, -33.83, 6, "Rhodes"),
    (151.21, -33.91, 6, "Zetland"),
    (151.15, -33.95, 6, "Wolli Creek"),
    (151.07, -33.83, 6, "Wentworth Point"),
    # Semi-Rural Lifestyle Dweller — urban fringe acreage
    (151.03, -33.69, 7, "Dural"),
    (151.02, -33.67, 7, "Galston"),
    (150.70, -34.00, 7, "Cobbitty"),
    (150.72, -33.88, 7, "Kemps Creek"),
]

ethnicity_tags = {
    "Newtown":        "Anglo/Creative",
    "Surry Hills":    "Anglo/Creative",
    "Marrickville":   "Greek/Vietnamese",
    "Blacktown":      "Filipino/Multicultural",
    "Rooty Hill":     "Filipino/Pacific Islander",
    "Liverpool":      "Lebanese/Iraqi/Indian",
    "Parramatta":     "Indian/Chinese",
    "Cabramatta":     "Vietnamese",
    "Fairfield":      "Vietnamese/Iraqi/Khmer",
    "Harris Park":    "Indian",
    "Lakemba":        "Lebanese/Bangladeshi",
    "Auburn":         "Turkish/Lebanese",
    "Bankstown":      "Lebanese/Multicultural",
    "Rockdale":       "Nepalese/Chinese",
    "Campsie":        "Chinese/Korean",
    "Strathfield":    "Korean/Chinese",
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
    "Stanhope Gardens": "Filipino/Indian",
    "The Ponds":         "Filipino/Chinese/Indian",
    "Rouse Hill":        "Filipino/Indian/Chinese",
    "Castle Hill":       "Chinese/Indian/Filipino",
    "Kellyville":     "Chinese/Indian",
    "Baulkham Hills": "Chinese/Filipino",
    "Chatswood":      "Chinese/Korean",
    "Ryde":           "Chinese/Multicultural",
    "Epping":            "Chinese/Indian",
    "Vaucluse":          "Anglo-Australian/European",
    "Bellevue Hill":     "Anglo-Australian/Jewish",
    "Mosman":            "Anglo-Australian",
    "Double Bay":        "Anglo-Australian/European",
    "Rhodes":            "Chinese/Mandarin",
    "Zetland":           "Chinese/Indian",
    "Wolli Creek":       "Chinese/Korean",
    "Wentworth Point":   "Chinese/Mandarin",
    "Dural":             "Anglo-Australian",
    "Galston":           "Anglo-Australian",
    "Cobbitty":          "Anglo-Australian/Italian",
    "Kemps Creek":       "Anglo-Australian/Lebanese",
}

FEATURES = ["longitude", "latitude", "median_age", "weekly_income",
            "pct_born_overseas", "pct_renting", "pct_english_only", "pct_university"]

# ── Generate training data & train model (runs once at startup) ───
np.random.seed(42)

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
X  = df[FEATURES].values
y  = df["subculture_id"].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

knn_full = KNeighborsClassifier(n_neighbors=5)
knn_full.fit(X_scaled, y)

print(f"Model ready — test accuracy: {accuracy_score(y_test, knn.predict(X_test))*100:.1f}%")

# ── Helpers ───────────────────────────────────────────────────────
_centroid_lon = sum(lon for lon, lat, _, _ in suburb_anchors) / len(suburb_anchors)
_centroid_lat = sum(lat for lon, lat, _, _ in suburb_anchors) / len(suburb_anchors)

def lookup_suburb(name):
    name_l = name.strip().lower()
    if not name_l:
        return _centroid_lon, _centroid_lat, "no suburb entered (centroid used)"
    for lon, lat, _, suburb in suburb_anchors:
        if suburb.lower() == name_l:
            return lon, lat, suburb
    for lon, lat, _, suburb in suburb_anchors:
        if name_l in suburb.lower() or suburb.lower() in name_l:
            return lon, lat, suburb
    return _centroid_lon, _centroid_lat, "unrecognised suburb (centroid used)"


def run_prediction(form):
    suburb_input = form.get("suburb", "").strip()
    age    = float(form.get("age",    30))
    income = float(form.get("income", 800))
    born   = form.get("born_overseas",  "no")
    lang   = form.get("other_language", "no")
    rent   = form.get("renting",        "no")
    uni    = form.get("university",     "no")

    lon, lat, matched_suburb = lookup_suburb(suburb_input)

    feature_values = [
        lon,
        lat,
        age,
        income,
        72.0  if born == "yes" else 18.0,
        72.0  if rent == "yes" else 18.0,
        22.0  if lang == "yes" else 80.0,
        55.0  if uni  == "yes" else 20.0,
    ]

    query_arr    = np.array([feature_values])
    query_scaled = scaler.transform(query_arr)

    pred_id   = knn.predict(query_scaled)[0]
    pred_name = subcultures[pred_id]

    distances, idxs = knn_full.kneighbors(query_scaled, return_distance=True)
    distances = distances[0]
    idxs      = idxs[0]

    neighbours = []
    for dist, idx in zip(distances, idxs):
        suburb = df.iloc[idx]["suburb"]
        neighbours.append({
            "suburb":    suburb,
            "community": ethnicity_tags.get(suburb, "—"),
            "distance":  round(float(dist), 3),
        })

    suburb_breakdown = Counter(n["suburb"] for n in neighbours).most_common()

    neighbor_scaled = X_scaled[idxs]
    per_feature_sq  = (query_scaled - neighbor_scaled) ** 2
    mean_sq         = per_feature_sq.mean(axis=0)
    total           = mean_sq.sum()
    pct_arr         = (mean_sq / total * 100) if total > 0 else np.zeros(len(FEATURES))

    order = np.argsort(pct_arr)[::-1]
    max_val = pct_arr[order[0]]
    feature_contributions = [
        {
            "name":    FEATURES[i],
            "pct":     round(float(pct_arr[i]), 1),
            "bar_pct": round(float(pct_arr[i] / max_val * 100), 1) if max_val > 0 else 0,
        }
        for i in order
    ]

    return dict(
        pred_name=pred_name,
        pred_class=PRED_CLASS[pred_name],
        pred_description=PRED_DESCRIPTION[pred_name],
        neighbours=neighbours,
        suburb_breakdown=suburb_breakdown,
        feature_contributions=feature_contributions,
        profile=dict(
            suburb_input=suburb_input or "Not specified",
            matched_suburb=matched_suburb,
            age=int(age),
            income=int(income),
            born_overseas=born,
            other_language=lang,
            renting=rent,
            university=uni,
        ),
    )

# ── Routes ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", active_page="home")


@app.route("/quiz")
def quiz():
    return render_template("quiz.html", active_page="quiz")


@app.route("/how-it-works")
def how_it_works():
    return render_template("how-it-works.html", active_page="how")


@app.route("/predict", methods=["POST"])
def predict():
    result = run_prediction(request.form)
    return render_template("results.html", active_page="quiz", **result)


if __name__ == "__main__":
    app.run(debug=True)
