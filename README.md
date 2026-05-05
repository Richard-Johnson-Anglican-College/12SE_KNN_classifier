# 🗺️ Sydney Subculture Classifier

> A K-Nearest Neighbours machine learning project that classifies Greater Sydney residents into distinct demographic subcultures — built as a Year 12 Software Engineering project at Richard Johnson Anglican College.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0.3-000000?style=flat-square&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.0-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## 🎯 What Does It Do?

Answer 7 questions about yourself — your suburb, age, income, and a few yes/no lifestyle questions — and this app uses a trained **K-Nearest Neighbours (KNN)** model to predict which of **8 Sydney subcultures** you belong to.

It then shows you *exactly why* by revealing your 5 closest demographic neighbours, the communities they represent, and a feature contribution chart breaking down which factors drove the prediction.

---

## 🏙️ The 8 Sydney Subcultures

| # | Subculture | Typical Suburbs | Profile |
|---|-----------|-----------------|---------|
| 1 | **Inner-City Creative** | Newtown, Surry Hills, Marrickville | Young renters in creative, academic & hospitality industries |
| 2 | **Aspirational Westie** | Blacktown, Liverpool, Parramatta, Penrith | Working/middle-class families in the outer west |
| 3 | **Cultural Enclave** | Cabramatta, Lakemba, Auburn, Hurstville | High overseas-born communities, strong ethnic identity |
| 4 | **Surf-Urbanite** | Manly, Cronulla, Mona Vale, Hornsby | Older, high-income coastal owner-occupiers |
| 5 | **Diaspora Professional** | Castle Hill, Chatswood, Epping, Ryde | High-income university-educated overseas-born homeowners |
| 6 | **Harbor Elite** | Vaucluse, Mosman, Double Bay, Bellevue Hill | Ultra-high-income professionals in Eastern Suburbs |
| 7 | **Vertical Cosmopolitan** | Rhodes, Zetland, Wolli Creek | Young apartment-dwellers in high-density rail corridors |
| 8 | **Semi-Rural Lifestyle Dweller** | Dural, Galston, Cobbitty | Tradespeople & business owners on the urban fringe |

---

## ✨ Features

- **7-question web quiz** — suburb, age, income, birthplace, home language, renting status, and university education
- **8-class KNN classifier** trained on 1,530+ synthetic census-style data points across 52 Greater Sydney suburbs
- **Diagnostic results page** — prediction, confidence class, 5 nearest neighbours with suburb + community + Euclidean distance
- **Feature contribution chart** — per-feature squared distance breakdown showing what drove the prediction
- **Fuzzy suburb matching** — partial name matching with Sydney CBD fallback
- **99%+ model accuracy** on held-out test data

---

## 🗂️ Project Structure

```
12SE_KNN_classifier/
│
├── app.py                  # Flask web app — model training, routes & prediction logic
├── knn_census.py           # Standalone script: multi-dimensional KNN with diagnostic report
├── knn_interactive.py      # CLI tool: live 7-question interview → full diagnostic output
├── visualize_gws.py        # 2D spatial KNN visualiser plotted over Sydney map
├── generate_map_data.py    # Utility: generates suburb coordinate data
├── fake_census.csv         # Synthetic census dataset (1530 rows × 8 features)
├── sydney_dark.png         # Dark-themed Sydney map for spatial visualisation
├── requirements.txt        # Python dependencies
│
├── templates/              # Flask Jinja2 HTML templates
│   ├── base.html           # Shared layout & navigation
│   ├── index.html          # Landing page
│   ├── quiz.html           # 7-question prediction form
│   ├── results.html        # Dynamic results & diagnostics
│   └── how-it-works.html   # ML/KNN explainer page
│
├── UI_test/                # Static HTML/CSS mockup (no backend)
│   ├── index.html
│   ├── quiz.html
│   ├── results.html
│   ├── how-it-works.html
│   └── style.css
│
└── static/                 # CSS & static assets for Flask app
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Richard-Johnson-Anglican-College/12SE_KNN_classifier.git
cd 12SE_KNN_classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Flask web app
python app.py
```

Then open **http://localhost:5000** in your browser.

### Run the CLI Tools

```bash
# Interactive CLI — answer questions in your terminal
python knn_interactive.py

# Standalone diagnostic report with hardcoded test profile
python knn_census.py

# 2D spatial visualiser plotted over Sydney map
python visualize_gws.py
```

---

## 🧠 How the Model Works

### Algorithm: K-Nearest Neighbours (K=5)

KNN classifies a new data point by finding the **5 most similar** data points in the training set (its "neighbours") and taking a majority vote on their class labels.

```
New Person → Scale Features → Find 5 Nearest Neighbours → Majority Vote → Predicted Subculture
```

### The 8 Features

Each "resident" in the dataset is described by 8 census-style features:

| Feature | Description |
|---------|-------------|
| `longitude` | Geographic longitude |
| `latitude` | Geographic latitude |
| `median_age` | Median age of suburb residents |
| `weekly_income` | Median weekly household income ($) |
| `pct_born_overseas` | % of residents born overseas |
| `pct_renting` | % of residents who rent |
| `pct_english_only` | % speaking only English at home |
| `pct_university` | % with a university degree |

### Personal-to-Census Proxy Mapping

Because the quiz asks personal yes/no questions but the model expects suburb-level census percentages, answers are mapped to representative proxy values:

| Quiz Question | Yes → | No → |
|--------------|-------|------|
| Born overseas? | `pct_born_overseas` = 72 | 18 |
| Other language at home? | `pct_english_only` = 22 | 80 |
| Renting? | `pct_renting` = 62 | 28 |
| University degree? | `pct_university` = 55 | 20 |

### Feature Standardisation

All features are normalised using `StandardScaler` (zero mean, unit variance) before training. This prevents high-magnitude features like `weekly_income` from dominating percentage-based features like `pct_university`.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Training samples | ~1,224 (80% split) |
| Test samples | ~306 (20% split) |
| K (neighbours) | 5 |
| Test accuracy | **≥ 99%** |
| Suburb anchors | 52 across Greater Sydney |

---

## 🗺️ Development Roadmap

- [x] **Part 1** — 2D spatial KNN visualiser (`visualize_gws.py`)
- [x] **Part 1** — 8-feature multi-dimensional census model (`knn_census.py`)
- [x] **Part 1** — Interactive CLI with nearest-neighbour diagnostics (`knn_interactive.py`)
- [x] **Part 2** — Full static UI mockup (`UI_test/`)
- [x] **Part 3** — Flask web app wiring quiz → model → dynamic results (`app.py`)
- [ ] **Future** — Replace synthetic data with real ABS QuickStats census data

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | Flask 3.0.3 |
| Machine learning | scikit-learn 1.6.0 |
| Data manipulation | pandas 2.2.2, NumPy 2.1.0 |
| Visualisation | matplotlib 3.9.2 |
| Frontend | HTML5, CSS3, Jinja2 |

---

## 📄 Licence

This project was created for educational purposes as part of the NSW Year 12 Software Engineering course at **Richard Johnson Anglican College**.
