# Sydney Subculture Prediction: Project Specification & Roadmap

## Overview
This project applies the K-Nearest Neighbors (KNN) machine learning algorithm to classify synthetic demographic and geographic data into eight distinct Sydney subcultures:
1. **Inner-City Creative**
2. **Aspirational Westie**
3. **Cultural Enclave**
4. **Surf-Urbanite**
5. **Diaspora Professional**
6. **Harbor Elite**
7. **Vertical Cosmopolitan**
8. **Semi-Rural Lifestyle Dweller**

What started as a basic 2D spatial demonstration has evolved into an 8-feature socio-economic classification model with an interactive prediction tool, a full Flask web application, and a beginner-friendly educational explainer.

---

## Part 1: Completed

### 1. Spatial KNN Visualizer (`visualize_gws.py`)
- **Core Function:** A 2D spatial KNN model that predicts a subculture purely based on GPS coordinates (Longitude/Latitude).
- **Data Generation:** Defined 26 specific "anchor" suburbs across Greater Sydney (from Bondi to the Blue Mountains) and assigned each to one of the 4 subcultures. Generated synthetic "residents" by adding geographic jitter around these anchors.
- **Visualization:** Mapped data points over a dark-themed map of Sydney (`sydney_dark.png`), correctly calibrating the bounding box (`150.43` to `151.40` Longitude, `-34.18` to `-33.40` Latitude) to ensure points align accurately with real-world map labels.
- **Limitation:** The spatial model is highly deterministic and manual. It "memorizes" regions we defined rather than discovering patterns in complex data.

### 2. Multi-Dimensional Census Model (`knn_census.py`)
- **Core Function:** Upgraded the model to use 8 socio-economic features to classify residents, mimicking real ABS Census data:
  - `longitude` & `latitude`
  - `median_age`
  - `weekly_income`
  - `pct_born_overseas`
  - `pct_renting`
  - `pct_english_only`
  - `pct_university`
- **Data Generation:** Created unique statistical profiles (means and standard deviations) for each subculture. Generated 1,470 synthetic residents (49 suburbs × 30 records each) based on these profiles, anchored around 49 Greater Sydney suburbs (each suburb also tagged with dominant ethnicities).
- **Machine Learning:**
  - Standardized the features using `StandardScaler` to prevent larger numbers (like income) from dominating percentages.
  - Achieved 100% accuracy on a held-out 20% test split (reflects clean separation of synthetic data, not real-world performance).
  - Demonstrated that the model can classify a "mystery person" based purely on their socio-economic profile, proving the KNN is evaluating complex multi-dimensional patterns, not just map coordinates.

### 3. Diagnostic Report & Nearest Neighbour Exposure (`knn_census.py`)
- **Nearest Neighbours:** Added `ethnicity_tags` dict mapping all 49 suburbs to their dominant ethnic communities. A second `knn_full` instance is fit on the entire dataset and `knn_full.kneighbors()` is called to extract the 5 closest residents for any given profile.
- **Output:** The script now prints exactly *who* the model grouped the mystery person with — suburb name, ethnic community, and Euclidean distance for each of the 5 neighbours, plus a grouped suburb breakdown (e.g., `Campsie×2, Cabramatta×1…`).
- **Feature Contribution Chart:** Per-feature squared distance between the mystery person and their neighbours is computed, normalised to percentages, and rendered as an ASCII bar chart — showing which features drove the prediction (e.g., `pct_english_only` at 31.6%).

### 4. Interactive Prediction Tool (`knn_interactive.py`)
- **Core Function:** A standalone CLI tool that replaces the hardcoded mystery person with a live 7-question interview. Trains the same model silently, then collects user input and runs the full diagnostic report.
- **Personal-to-Census Proxy Mapping:** Because the model expects suburb-level census aggregates, personal yes/no answers are mapped to representative numerical values:
  - Born overseas → `pct_born_overseas`: Yes = 72, No = 18
  - Other language at home → `pct_english_only`: Yes = 22, No = 80
  - Renting → `pct_renting`: Yes = 72, No = 18
  - University degree → `pct_university`: Yes = 55, No = 20
  - Age and income are passed through directly.
  - Suburb name is fuzzy-matched against 49 anchor suburbs for coordinates; falls back to the geographic centroid of all suburbs (~Parramatta area) when blank or unrecognised.
- **Output:** Full diagnostic report identical to `knn_census.py` — prediction, 5 nearest neighbours with distances, suburb breakdown, and ASCII feature contribution chart.

---

## Part 2: Completed

### 1. Web UI Mockup (`UI_test/`)
A static 4-page HTML/CSS mockup built for UI approval, styled as a modern Australian Government (NSW DTA) website. No backend — `results.html` uses hardcoded example data (Newtown / Inner-City Creative run).

| File | Purpose |
|---|---|
| `index.html` | Landing page — project intro, 8 subculture image cards, CTA |
| `quiz.html` | 7-question form with suburb input and yes/no toggle buttons |
| `results.html` | Diagnostic report — result banner, neighbours table, feature contribution bar chart |
| `how-it-works.html` | Beginner-friendly ML/KNN explainer, proxy mapping table, model specs |

---

## Part 3: Completed

### 1. Flask Web Application (`app.py`)
Full Flask web app connecting the quiz form to the KNN model with dynamic results rendering.

| Route | Purpose |
|---|---|
| `/` | Home page — 8 subculture image cards with colour-coded borders |
| `/quiz` | 7-question prediction form |
| `/predict` | POST handler — runs KNN, renders results with neighbours + feature chart |
| `/how-it-works` | Beginner-friendly KNN/ML explainer |

**Key implementation details:**
- Model trains from scratch on every server start using `np.random.seed(42)` for determinism
- `StandardScaler` fitted on training split only, applied to both training and query vectors
- Two KNN instances: `knn` (80% train split, for accuracy measurement) and `knn_full` (all 1,470 records, for neighbour lookup)
- Fuzzy suburb matching with geographic centroid fallback
- Responsive CSS grid: 4-column desktop / 2-column tablet / 1-column mobile
- Subculture hero banner with colour theming and photo image per result
- Feature contribution chart showing per-feature share of mean squared distance

---

## Part 4: Future (Optional)

### Transition to Real Data
Replace the `fake_census.csv` generation script with a pipeline that ingests actual ABS QuickStats data for Greater Sydney suburbs, allowing the KNN model to discover genuine, unprompted demographic clusters.
