import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# ---------------------------------------------------------------
# SUBCULTURE DEFINITIONS (4 groups, real suburb coordinates)
# ---------------------------------------------------------------
subcultures = {
    0: "Inner-City Creative",
    1: "Aspirational Westie",
    2: "Cultural Enclave",
    3: "Surf-Urbanite",
}

colors = {
    0: '#9b59b6',   # Purple  - Inner-City Creative
    1: '#e67e22',   # Orange  - Aspirational Westie
    2: '#27ae60',   # Green   - Cultural Enclave
    3: '#2980b9',   # Blue    - Surf-Urbanite
}

# Suburb anchor points: (longitude, latitude, subculture_id, name, ethnicity)
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
    (150.55, -33.71, 1, "Springwood",     "Anglo-Australian / Alternative"),
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
# GENERATE SYNTHETIC TRAINING DATA (jitter around each suburb)
# ---------------------------------------------------------------
np.random.seed(42)
POINTS_PER_SUBURB = 25   # how many "residents" per suburb
SPREAD = 0.025           # how far they spread out (in degrees ~2.5km)

train_lons, train_lats, train_labels = [], [], []

for lon, lat, label, name, ethnicity in suburb_anchors:
    jitter_lon = np.random.normal(lon, SPREAD, POINTS_PER_SUBURB)
    jitter_lat = np.random.normal(lat, SPREAD, POINTS_PER_SUBURB)
    train_lons.extend(jitter_lon)
    train_lats.extend(jitter_lat)
    train_labels.extend([label] * POINTS_PER_SUBURB)

X_train = np.column_stack([train_lons, train_lats])
y_train = np.array(train_labels)

# ---------------------------------------------------------------
# MYSTERY PERSON — change these coords to predict anyone's subculture
# ---------------------------------------------------------------
mystery_lon = 151.28
mystery_lat = -33.80
mystery_point = np.array([[mystery_lon, mystery_lat]])

# ---------------------------------------------------------------
# TRAIN KNN & PREDICT
# ---------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

prediction_index = knn.predict(mystery_point)[0]
prediction_name = subcultures[prediction_index]

# ---------------------------------------------------------------
# LOAD BACKGROUND MAP
# ---------------------------------------------------------------
img = mpimg.imread('sydney_dark.png')

# Bounding box recalibrated to full extent of sydney.png:
# Left edge  ~Warrimoo/Winmalee:   150.43
# Right edge ~Eastern coast:        151.40
# Top edge   ~Glenorie:            -33.40
# Bottom edge~Camden/The Oaks:     -34.18
LON_MIN, LON_MAX = 150.43, 151.40
LAT_MIN, LAT_MAX = -34.18, -33.40

# ---------------------------------------------------------------
# DRAW THE CHART
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 10))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

# Background map
ax.imshow(img, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
          aspect='auto', alpha=0.75, zorder=0)

# Plot each subculture group
for group_id, group_name in subcultures.items():
    mask = y_train == group_id
    ax.scatter(X_train[mask, 0], X_train[mask, 1],
               color=colors[group_id],
               s=40, alpha=0.6, edgecolors='white', linewidths=0.3,
               zorder=2)

# Plot suburb anchor labels (show suburb + ethnicity)
for lon, lat, label, name, ethnicity in suburb_anchors:
    ax.plot(lon, lat, marker='s', color=colors[label],
            markersize=10, markeredgecolor='white', markeredgewidth=1.2, zorder=3)
    ax.text(lon + 0.01, lat + 0.012, f"{name}\n{ethnicity}",
            fontsize=7, color='white', fontweight='bold',
            bbox=dict(facecolor=colors[label], alpha=0.80, edgecolor='none',
                      boxstyle='round,pad=0.2'), zorder=4)

# Plot mystery person
ax.scatter(mystery_lon, mystery_lat, color='white', marker='*',
           s=600, edgecolors='black', linewidths=1.5,
           label='Mystery Person (Who am I?)', zorder=5)

# Neighbour radius circle
circle = plt.Circle((mystery_lon, mystery_lat), 0.06,
                    color='white', fill=False, linestyle='--',
                    linewidth=1.8, zorder=4)
ax.add_patch(circle)

# Prediction label
pred_color = colors[prediction_index]
ax.text(mystery_lon + 0.03, mystery_lat + 0.03,
        f">> {prediction_name}",
        fontsize=13, fontweight='bold', color='white',
        bbox=dict(facecolor=pred_color, alpha=0.95,
                  edgecolor='white', boxstyle='round,pad=0.5'),
        zorder=6)

# ---------------------------------------------------------------
# LEGEND
# ---------------------------------------------------------------
legend_handles = [
    mpatches.Patch(color=colors[i], label=f'{subcultures[i]}') for i in subcultures
]
legend_handles.append(
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='white',
               markersize=14, label='Mystery Person', linestyle='None')
)
ax.legend(handles=legend_handles, loc='lower right',
          framealpha=0.85, fontsize=10,
          facecolor='#1a1a2e', labelcolor='white', edgecolor='white')

# ---------------------------------------------------------------
# AXIS LABELS & TITLE
# ---------------------------------------------------------------
ax.set_xlim(LON_MIN, LON_MAX)
ax.set_ylim(LAT_MIN, LAT_MAX)

# Axis tick labels — suburb names at the matching lon/lat positions
lon_ticks = [150.50, 150.65, 150.80, 150.98, 151.15, 151.32]
lon_labels = ["Blue Mtns /\nSpringwood", "Penrith /\nSt Marys", "Mt Druitt /\nBlacktown", "Parramatta", "Inner West", "Eastern /\nBeaches"]
ax.set_xticks(lon_ticks)
ax.set_xticklabels(lon_labels, fontsize=8, color='white')

lat_ticks = [-34.10, -33.93, -33.80, -33.68, -33.53]
lat_labels = ["Campbelltown\n(South)", "Liverpool /\nCronulla", "Cabramatta /\nBankstown", "Parramatta /\nMarrickville", "Hornsby /\nMona Vale"]
ax.set_yticks(lat_ticks)
ax.set_yticklabels(lat_labels, fontsize=8, color='white')

ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('white')

ax.set_title(
    "Sydney Subculture Predictor (KNN)\n"
    "\"Drop a pin — KNN tells you what subculture lives there\"",
    fontsize=14, fontweight='bold', color='white', pad=15
)
ax.grid(True, linestyle='--', alpha=0.15, color='white', zorder=1)

plt.tight_layout()
plt.savefig('knn_sydney.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Mystery person at ({mystery_lon}, {mystery_lat})")
print(f"KNN Prediction: '{prediction_name}'")
