# 📺 STC TV Data Analysis: Viewing Behavior, Forecasting & Recommendations

An end-to-end analysis of STC (Jawwy) TV viewing data, covering three tasks: **what users watch**, **where watch time is heading**, and **what to recommend to each viewer**.

Completed as part of the Misk Skills × stc data analyst virtual experience.

## Repository structure

```
├── notebooks/
│   ├── StcTV_User_Analysis.ipynb        # Task 1: viewing behavior analysis
│   ├── Viewing_Prediction_Model.ipynb   # Task 2: watch-hours forecasting (Prophet)
│   └── stc_TV_Task3_Recommender.ipynb   # Task 3: recommender (KNN + cosine)
├── requirements.txt
└── README.md
```

## Data

The datasets are provided by the program and are **not included** in this repository. Each notebook reads its own file:

| Task | Dataset | Description |
|---|---|---|
| 1 | `stc TV Data Set_T1.xlsb` | 1,048,575 viewing records, 13 columns (user, program, duration, class, HD flag, genre...) |
| 2 | `stc_TV_Data_Set_T2.csv` | 86 days of daily total watch hours (Jan–Apr 2018) |
| 3 | `stc TV Data Set_T3.xlsx` | 1,048,575 records: user, program, rating (1–4), date, genre |

Update the file paths in the first cells of each notebook before running (Tasks 1 and 2 were written for Google Colab with Google Drive).

---

## Task 1: User viewing behavior

**Questions:** which programs are watched most, how do series compare with movies, and how does HD relate to viewing?

**Key findings**
- **Series create time, movies create reach.** Series account for 255,098 watch hours (71%) from 3,901 users, while movies reach 11,355 users but account for 103,444 hours.
- **Animation leads.** 6 of the top 10 programs by watch time are animated movies. *The Boss Baby* alone has 2,961 hours and 3,389 users.
- **HD reaches more users, SD holds the hours.** 11,000 users watched in HD (90,178 hours) versus 6,728 in SD (268,364 hours, about 75% of the total). The dataset cannot explain why; a breakdown by content type would be the next step.

**Method notes:** series episodes are separated by season and episode before ranking programs; missing values exist only in `program_desc`.

---

## Task 2: Forecasting watch hours

**Objective:** analyze viewing patterns and forecast future daily watch hours.

**Approach**
- EDA and seasonal decomposition (trend, weekly pattern, residuals)
- Prophet model tuned via grid search (40 parameter combinations)
- Evaluated with rolling-origin time series cross-validation (11 forecasts, 14-day horizon), not a static train/test split
- Best config: `changepoint_prior_scale=0.001`, `seasonality_prior_scale=1.0`, `weekly_seasonality=False`

**Results**

| Metric | Value |
|---|---|
| MAE | 75.11 hours (about 10% of the 781-hour daily average) |
| RMSE | 92.61 hours |

**Key insights**
- **Overall downward trend.** Average daily watch hours fall from 867 (Jan) to 823 (Feb), 755 (Mar) and 674 (Apr), about -22% from January to April. The 60-day forecast continues from 641 hours (May 1) to 507 (Jun 29).
- **The forecast is smoother than the actual data by design.** It captures trend, not day-to-day noise.
- **With weekly seasonality off, the forecast is a straight downward line,** so the "highest" and "lowest" forecast days are simply its first and last days.
- **A 60-day horizon is long** relative to 86 days of history, so uncertainty grows over time.

**Data limitation worth knowing:** the dataset contains **weekdays only**. All Saturdays and Sundays are missing (34 of the 120 calendar days from Jan 1 to Apr 30), which is the most likely reason weekly seasonality hurt cross-validation accuracy, alongside the short history (~12 weeks). It also means the decomposition with `period=7` in the EDA runs over rows that skip weekends, so the weekly pattern it shows should be treated with caution (a 5-row period would match the data).

**What I learned:** my first version scored the model on the same data it trained on, giving a misleadingly good MAE. Evaluating out-of-sample with `cross_validation` reflects real forecasting performance.

---

## Task 3: Recommendation system

**Objective:** recommend programs to viewers, and show the top 5 for people who watched *Moana*.

**Approach:** item-based collaborative filtering (KNN with cosine similarity). It never looks at titles, only at *who watched what*.

1. **One row per (user, program).** 608,338 of the 1,048,575 rows repeat a user + program pair, so I kept the highest rating: 1,048,575 → 440,237 rows.
2. **Drop rarely watched programs** (fewer than 50 viewers) to avoid chance similarities: 8,013 → 2,268 programs.
3. **Sparse program × user matrix** (2,268 × 11,532) so it fits comfortably in memory.
4. **Fit `NearestNeighbors(metric='cosine')`** and look up the closest programs.


| # | Program | Genre | Similarity |
|---|---|---|---|
| 1 | Trolls | Animation | 0.589 |
| 2 | Surf's Up: WaveMania | Animation | 0.539 |
| 3 | The Mermaid Princess | Animation | 0.502 |
| 4 | The Boss Baby | Animation | 0.459 |
| 5 | The Jetsons & WWE: Robo-WrestleMania! | Animation | 0.444 |

**Assumptions and limits:** `rating` appears to be a 4-level bucket of watch duration (inferred from comparing with the Task 1 data, not documented). Programs under 50 viewers cannot be recommended (cold start). Results were sanity-checked by genre but not yet evaluated offline (e.g. precision@k).

---

## Data quality findings

Noticed while working across the three datasets:

- **58% of records repeat a user + program pair** with no session ID to tell viewings apart.
- **Truncated program names.** The capital letter "S" disappears inside names (`The Amazing pider-Man`, `The murfs`, `Bedtime tories`).
- **4,751 records without a real genre** (`NOT_DEFINED_IN_UMS`, `SERIES_NOT_ADDED_UNDER_ANY_GENRE`).
- **Weekends missing** from the Task 2 data (see above).
- **Excel serial dates** in Task 1 need `pd.to_datetime(..., unit='d', origin='1899-12-30')` to convert correctly.

## Tech stack

Python, pandas, NumPy, SciPy, scikit-learn, Prophet, statsmodels, Plotly, Matplotlib

## Run it

```bash
pip install -r requirements.txt
```

Open a notebook in `notebooks/`, update the dataset path, and run all cells.
