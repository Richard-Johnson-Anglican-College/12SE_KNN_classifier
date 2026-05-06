# Explaining KNN — A Beginner's Guide

> This document is a companion to the Sydney Subculture Classifier project.
> It is written for students who are new to machine learning and K-Nearest Neighbours.

---

## What is Machine Learning?

Traditional computer programs follow rules a programmer writes explicitly:

> *"If income is high and suburb is coastal, label it Surf-Urbanite."*

The problem is that real-world patterns are too complex and numerous to hand-code.

**Machine learning flips this:** instead of writing the rules, you give the program hundreds of labelled examples and let it *discover* the patterns itself. The program **learns** from data.

This tool uses a machine learning model trained on 1,470 synthetic Sydney residents — each one described by 8 demographic numbers and pre-labelled with one of eight subcultures. The model learned which combinations of numbers correspond to which group, so it can now classify people it has never seen before.

---

## What is K-Nearest Neighbours (KNN)?

KNN is one of the most intuitive machine learning algorithms. Its logic mirrors how humans think:

> **"You are most like the people closest to you."**

Imagine you have moved to a new suburb and want to know what kind of area it is. You look at your five nearest neighbours — what they earn, whether they rent, where they were born — and make a judgement based on what most of them have in common. KNN does exactly this, but with maths instead of intuition.

Each resident in the database is plotted as a point in an **8-dimensional space** (one dimension per feature). When you submit your answers, your profile becomes a new point in that same space. The model measures the straight-line distance from your point to every other point, finds the 5 closest, and lets them vote. The subculture with the most votes wins.

---

## What Happens Step by Step

### Step 1 — Your answers become 8 numbers

Each answer is converted into a numerical feature. For example:
- `"Yes, I rent"` → `pct_renting = 72`
- Your suburb name → `longitude` and `latitude` coordinates

### Step 2 — Features are standardised

Income ($850/wk) and a percentage (72%) are on very different scales. If left raw, income would dominate every distance calculation just because it is a bigger number — **not because it is more important.**

`StandardScaler` rescales all 8 features to the same range (mean = 0, roughly ±1–2) so every feature competes fairly in the distance formula.

### Step 3 — Distance to all 1,470 residents is calculated

The model computes the straight-line (Euclidean) distance between your 8-number profile and every resident's 8-number profile — the same formula as Pythagoras, extended to 8 dimensions:

```
distance = √( (age₁-age₂)² + (income₁-income₂)² + (born₁-born₂)² + ... )
```

Because all features have been scaled, the result is a **dimensionless number** — not kilometres, dollars, or years. A distance of roughly 1.0 means the two profiles are about one standard deviation apart across all features combined.

**Rough distance guide:**

| Distance | What it means |
|---|---|
| < 1.0 | Very close match — nearly identical profile |
| 1–2 | Reasonable match — a few features differ noticeably |
| 2–3 | Weaker match — meaningful differences across several features |
| > 3 | Poor match — crossing into different demographic territory |

### Step 4 — The 5 nearest neighbours vote

The 5 residents with the smallest distance are identified. Each casts one vote for their subculture. The subculture with the most votes becomes your prediction.

### Step 5 — Feature contributions are explained

For each feature, the model measures how much of the mean squared distance between you and your neighbours it was responsible for. Features where you differed most from your neighbours had the biggest influence — shown as the longest bars in the results chart.

---

## What Does the k in KNN Mean?

The value *k* is a choice you make when building the model. It controls how many neighbours vote.

**Common misconception:** *"A higher k is better because more data = more accurate."*

More *training data* genuinely does improve accuracy — that is why this model uses 1,470 records instead of 147. But **k is different**: k is not how much data you train on, it is how many of those trained points you consult at *prediction time*.

A higher k means consulting neighbours that are **further away**, which dilutes the signal from the people who are actually most like you:

| k value | Effect |
|---|---|
| **k = 1** | Only the single closest person decides. Very responsive to your exact profile, but one unusual data point can flip the result entirely. |
| **k = 5** | Five closest people vote. A single outlier is outvoted 4-to-1. More stable, still close enough to your profile to be meaningful. *This is what this model uses.* |
| **k = 50** | The 50 closest people vote — but the 50th person may be quite far from you in demographic space. You are now averaging over a wide region. It is like asking 50 people spread across all of Western Sydney what suburb you belong in, instead of asking the 5 people standing right next to you. |

---

## Why 1,470 Records and Not Fewer?

The database is structured as **49 suburbs × 30 records each = 1,470 total**.

With k = 5, you need enough records per suburb that all five nearest neighbours are drawn from *within* the correct subculture — not from an adjacent one. At 30 records per suburb, the local neighbourhood is dense enough to make this reliable.

At ~3 records per suburb (147 total), the five nearest neighbours would frequently cross class boundaries, making predictions noisy and unreliable.

---

## How Personal Answers Become Census Numbers

The model was trained on suburb-level census aggregates — percentages of a suburb's population. Since you are answering as an individual, your yes/no answers are mapped to **percentage proxies**:

| Question | Census feature | Yes → | No → |
|---|---|---|---|
| Born overseas? | `pct_born_overseas` | 72 | 18 |
| Other language at home? | `pct_english_only` | 22 (low) | 80 (high) |
| Do you rent? | `pct_renting` | 72 | 18 |
| University degree? | `pct_university` | 55 | 20 |
| Age | `median_age` | Direct value | — |
| Weekly income | `weekly_income` | Direct value ($AUD) | — |
| Suburb name | `longitude` + `latitude` | Matched to 49 anchor suburbs | Centroid fallback |

For example: answering "Yes, I was born overseas" sets `pct_born_overseas = 72`, meaning the model treats you as if you live in a suburb where 72% of residents were born overseas. These values were chosen to sit near the high and low ends of Sydney's real distribution so your answer places your profile in the correct region of the data.

---

## A Worked Example

**Input:** age 50, $500/wk, born in Australia, English at home, renting, no university, suburb: Blacktown

**Feature vector after proxy mapping:**

| Feature | Value |
|---|---|
| longitude | 150.906 (Blacktown) |
| latitude | -33.769 (Blacktown) |
| median_age | 50 |
| weekly_income | 500 |
| pct_born_overseas | 18 (born in Australia) |
| pct_renting | 72 (renting) |
| pct_english_only | 80 (English at home) |
| pct_university | 20 (no degree) |

**Why Aspirational Westie?**

- Location (Blacktown) anchors the point in outer-western Sydney
- Income ($500) is close to the Aspirational Westie average ($680)
- English at home (80%) rules out Cultural Enclave (25%) and Diaspora Professional (20%)
- The 5 nearest neighbours are all outer-western suburbs → all 5 votes go to Aspirational Westie

The model didn't find a perfect match — it found the *least wrong* match. KNN doesn't reason about what *should* fit; it measures distance to every point in the database and lets the 5 closest ones vote.

---

## Why 100% Accuracy?

The synthetic training data was generated from tightly defined statistical profiles — each subculture has deliberately distinct income, language, and rental patterns. This makes the eight groups very well-separated in 8D space, so KNN can classify them almost perfectly.

**Real-world census data would be messier and accuracy would be lower.** The 100% figure reflects how cleanly the synthetic data was generated, not real-world performance.
