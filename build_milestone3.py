"""
Bangkok Rail Explorer - ICT324 Milestone 3 builder
---------------------------------------------------
Input (same folder):
  drt2566_02.csv                          DRT station-level passenger data (data.go.th / datagov.mot.go.th)
  Bangkok_Rail_Explorer_Teletubbies.html  the Milestone 2 webpage

Output (folder "m3_output"):
  index.html            webpage with Milestone 2 fixes + new "Progress" section (real numbers)
  cleaned_data.csv      cleaned station-day table
  cleaning_log.csv      every cleaning step, why, rows affected
  model_results.csv     10-fold CV precision / recall / F1 / AUC per model

Run in Google Colab:
  1. Upload this file, the CSV and the HTML.
  2. !python build_milestone3.py
"""
import html, re, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

CSV_IN = Path("drt2566_02.csv")
HTML_IN = Path("Bangkok_Rail_Explorer_Teletubbies.html")
OUT = Path("m3_output")
OUT.mkdir(exist_ok=True)

ARL_NAMES = {"A1": "Suvarnabhumi", "A2": "Lat Krabang", "A3": "Ban Thap Chang", "A4": "Hua Mak",
             "A5": "Ramkhamhaeng", "A6": "Makkasan", "A7": "Ratchaprarop", "A8": "Phaya Thai"}

log = []
def step(name, why, affected):
    log.append({"step": name, "why": why, "rows_affected": int(affected)})
    print(f"[clean] {name}: {affected}")

# ------------------------------------------------------------------ 1. LOAD
def read_csv(path):
    for enc in ("utf-8-sig", "cp874", "tis-620"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str)
        except UnicodeDecodeError:
            continue
    sys.exit("Could not read the CSV encoding.")

if not CSV_IN.exists() or not HTML_IN.exists():
    sys.exit(f"Put {CSV_IN} and {HTML_IN} in this folder first.")

raw = read_csv(CSV_IN)
raw.columns = [c.strip().replace("\ufeff", "") for c in raw.columns]
colmap = {}
for c in raw.columns:
    k = c.lower().replace(" ", "_").replace(".", "")
    if k in ("no", "no_"): colmap[c] = "No"
    elif "arriv" in k: colmap[c] = "Arrival"
    elif "depart" in k: colmap[c] = "Departure"
    elif "station_code" in k or k == "code": colmap[c] = "Code"
    elif "station_name" in k or k == "name": colmap[c] = "Name"
    elif k == "date" or "date" in k: colmap[c] = "Date"
    elif k == "line": colmap[c] = "Line"
    elif k == "type": colmap[c] = "Type"
df = raw.rename(columns=colmap)
need = {"Code", "Arrival", "Departure", "Date"}
if not need.issubset(df.columns):
    sys.exit(f"Missing columns. Found: {list(raw.columns)}")
n_raw = len(df)
print(f"Loaded {n_raw} rows, columns: {list(raw.columns)}")

# ------------------------------------------------------------------ 2. CLEAN
if "No" in df.columns:
    df = df.drop(columns="No")
    step("Dropped 'No.' column", "Row number only, carries no information", 0)

before = len(df)
df = df.drop_duplicates()
step("Removed exact duplicate rows", "Same record listed twice would double-count passengers", before - len(df))

for c in ("Code", "Name", "Line", "Type"):
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()
df["Code"] = df["Code"].str.upper().str.replace(" ", "", regex=False)
step("Trimmed spaces, made station codes upper-case", "'a6 ' and 'A6' must be the same station", 0)

def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")
for c in ("Arrival", "Departure"):
    df[c] = to_num(df[c])
bad_num = df[["Arrival", "Departure"]].isna().any(axis=1)
step("Removed rows with missing / non-numeric passenger counts", "Cannot compute a daily total without both values", bad_num.sum())
df = df[~bad_num]

neg = (df["Arrival"] < 0) | (df["Departure"] < 0)
step("Removed negative passenger counts", "Passenger counts cannot be below zero", neg.sum())
df = df[~neg]

def fix_year(s):  # Thai Buddhist year 25xx -> Christian year
    return re.sub(r"\b(25\d\d)\b", lambda m: str(int(m.group(1)) - 543), str(s))
date_txt = df["Date"].astype(str).str.strip()
buddhist = date_txt.str.contains(r"\b25\d\d\b", regex=True).sum()
date_txt = date_txt.map(fix_year)
parsed = pd.to_datetime(date_txt, errors="coerce", format="mixed")
if parsed.isna().mean() > 0.2:
    parsed = pd.to_datetime(date_txt, errors="coerce", dayfirst=True, format="mixed")
step("Converted Thai Buddhist years to Christian years", "Python dates need Christian years (2567 -> 2024)", buddhist)
df["Date"] = parsed.dt.normalize()
bad_date = df["Date"].isna()
step("Removed rows with unreadable dates", "A record without a date cannot be placed on the timeline", bad_date.sum())
df = df[~bad_date]

df["Total"] = df["Arrival"] + df["Departure"]
zero = df["Total"] == 0
step("Removed station-days with 0 passengers", "Every station shows 0 on these days: data not reported, not real demand", zero.sum())
df = df[~zero]

# Same station listed twice on one date while the day before is missing -> first copy is the missing day
first = df.duplicated(subset=["Code", "Date"], keep="last")
present = set(zip(df["Code"], df["Date"]))
fixable = first & np.array([(c, d - pd.Timedelta(days=1)) not in present for c, d in zip(df["Code"], df["Date"])])
df.loc[fixable, "Date"] = df.loc[fixable, "Date"] - pd.Timedelta(days=1)
step("Fixed mislabeled dates (moved first copy to the missing day before)",
     "A date appeared twice with different counts while the previous day was missing", fixable.sum())
dup_key = df.duplicated(subset=["Code", "Date"], keep="first")
step("Kept one record per station per day", "Conflicting duplicates for the same station and date", dup_key.sum())
df = df[~dup_key]

df["Station"] = df["Code"].map(ARL_NAMES)
if "Name" in df.columns:
    df["Station"] = df["Station"].fillna(df["Name"])
df["Station"] = df["Station"].fillna(df["Code"])
step("Added English station names (A1-A8)", "Source names are Thai; the website is in English", df["Code"].isin(ARL_NAMES).sum())

df = df.sort_values(["Code", "Date"]).reset_index(drop=True)
q1 = df.groupby("Code")["Total"].transform(lambda s: s.quantile(.25))
q3 = df.groupby("Code")["Total"].transform(lambda s: s.quantile(.75))
outlier = df["Total"] > q3 + 3 * (q3 - q1)
df["Outlier_flag"] = outlier.astype(int)
step("Flagged extreme days (kept, not removed)", "Real events (e.g. free-fare days) - flagged so they can be explained", outlier.sum())

n_clean = len(df)
df.to_csv(OUT / "cleaned_data.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(log).to_csv(OUT / "cleaning_log.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------ 3. FEATURES + TARGET
# Target: is this a HIGH-demand day for this station? (above the station's own median)
df["High_demand"] = (df["Total"] > df.groupby("Code")["Total"].transform("median")).astype(int)

parts = []
for code, g in df.groupby("Code"):
    g = g.set_index("Date").asfreq("D")          # full calendar so lags are true previous days
    g["Code"] = code
    g["Lag1"] = g["Total"].shift(1)               # yesterday
    g["Lag7"] = g["Total"].shift(7)               # same weekday last week
    parts.append(g.reset_index())
m = pd.concat(parts).dropna(subset=["Total", "Lag1", "Lag7"])
m["DayOfWeek"] = m["Date"].dt.dayofweek
m["Weekend"] = (m["DayOfWeek"] >= 5).astype(int)
m["Month"] = m["Date"].dt.month
m["Year"] = m["Date"].dt.year
m["DayOfMonth"] = m["Date"].dt.day
m["High_demand"] = m["High_demand"].astype(int)
m = m.sort_values("Date").reset_index(drop=True)

cat = ["Code"]
num = ["DayOfWeek", "Weekend", "Month", "Year", "DayOfMonth", "Lag1", "Lag7"]
X, y = m[cat + num], m["High_demand"]
print(f"Modeling rows: {len(m)}  | positive class share: {y.mean():.2%}")

def prep(scale):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler() if scale else "passthrough", num)])

models = {
    "Baseline (majority class)": Pipeline([("p", prep(False)), ("m", DummyClassifier(strategy="most_frequent"))]),
    "Logistic Regression": Pipeline([("p", prep(True)), ("m", LogisticRegression(max_iter=2000))]),
    "Decision Tree": Pipeline([("p", prep(False)), ("m", DecisionTreeClassifier(max_depth=6, random_state=42))]),
    "Random Forest": Pipeline([("p", prep(False)), ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1))]),
}
scoring = {"precision": make_scorer(precision_score, zero_division=0),
           "recall": make_scorer(recall_score, zero_division=0),
           "f1": make_scorer(f1_score, zero_division=0),
           "roc_auc": "roc_auc"}

# ------------------------------------------------------------------ 4. 10-FOLD CV
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
rows = []
for name, pipe in models.items():
    r = cross_validate(pipe, X, y, cv=cv10, scoring=scoring)
    row = {"model": name}
    for k in scoring:
        row[k] = r[f"test_{k}"].mean(); row[k + "_sd"] = r[f"test_{k}"].std()
    rows.append(row)
    print(f"{name:28s} P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f} AUC={row['roc_auc']:.3f}")
res = pd.DataFrame(rows)
best = res.iloc[1:].sort_values("f1", ascending=False).iloc[0]

# Honesty check: train on past, test on future (no shuffling)
ts = cross_validate(models[best["model"]], X, y, cv=TimeSeriesSplit(n_splits=5), scoring=scoring)
ts_f1, ts_auc = ts["test_f1"].mean(), ts["test_roc_auc"].mean()
print(f"Time-split check ({best['model']}): F1={ts_f1:.3f} AUC={ts_auc:.3f}")

rf = models["Random Forest"].fit(X, y)
feat_names = list(rf.named_steps["p"].get_feature_names_out())
imp = pd.Series(rf.named_steps["m"].feature_importances_, index=feat_names)
imp.index = [i.split("__", 1)[1] for i in imp.index]
station_imp = imp[[i for i in imp.index if i.startswith("Code_")]].sum()
imp = imp[[i for i in imp.index if not i.startswith("Code_")]]
imp["Station"] = station_imp
top = imp.sort_values(ascending=False).head(4)
nice = {"Lag1": "yesterday's passengers", "Lag7": "same weekday last week", "DayOfWeek": "day of week",
        "Weekend": "weekend", "Month": "month", "Year": "year", "DayOfMonth": "day of month", "Station": "station"}

res.round(4).to_csv(OUT / "model_results.csv", index=False)

# ------------------------------------------------------------------ 5. WEBPAGE
e = html.escape
d0, d1 = df["Date"].min(), df["Date"].max()
lines = ", ".join(sorted(df["Line"].dropna().replace({"แอร์พอร์ต เรล ลิงก์": "Airport Rail Link"}).unique())) if "Line" in df.columns else "n/a"
n_st = df["Code"].nunique()
base = res.iloc[0]
gain = best["f1"] - base["f1"]
drop = best["f1"] - ts_f1

log_rows = "".join(
    f"<tr><td>{e(r['step'])}</td><td>{e(r['why'])}</td><td class='n'>{('column' if 'column' in r['step'] else 'none found') if r['rows_affected']==0 else format(r['rows_affected'], ',')}</td></tr>" for r in log)
res_rows = "".join(
    f"<tr{' class=best' if r['model']==best['model'] else ''}><td>{e(r['model'])}</td>"
    + "".join(f"<td class='n'>{r[k]:.3f}<small>±{r[k+'_sd']:.3f}</small></td>" for k in ("precision", "recall", "f1", "roc_auc"))
    + "</tr>" for _, r in res.iterrows())
top_txt = ", ".join(nice.get(i, i) for i in top.index)
n_beat = int((res.iloc[1:]["f1"] > base["f1"]).sum())
beat_txt = (f"All {len(res)-1} trained models beat the baseline; " if n_beat == len(res)-1
            else f"{n_beat} of {len(res)-1} models beat the baseline; ") + f"{best['model']} reaches AUC {best['roc_auc']:.3f}."

section = f"""
<section class="section m3" id="progress"><div class="sectionHead"><div><span class="kicker">05 / Milestone 3 progress</span><h2>Data cleaned, first models trained.</h2></div><p>Everything below is computed from the official DRT file. No simulated values are used in this section.</p></div>

<div class="m3Facts">
 <div><b>{n_raw:,}</b><span>raw rows</span></div>
 <div><b>{n_clean:,}</b><span>clean station-days</span></div>
 <div><b>{n_st}</b><span>stations ({e(lines)})</span></div>
 <div><b>{d0:%b %Y} – {d1:%b %Y}</b><span>date range</span></div>
</div>

<h3 class="m3H">1. Data cleaning: what we changed and why</h3>
<p class="m3P">Coverage audit result: the station-level DRT file covers the <strong>Airport Rail Link only</strong>. Other lines stay as demo data on the map until station-level files are found.</p>
<div class="m3Scroll"><table class="m3Table"><thead><tr><th>Change</th><th>Why</th><th class="n">Rows</th></tr></thead><tbody>{log_rows}</tbody></table></div>

<h3 class="m3H">2. First results: predicting a high-demand day</h3>
<p class="m3P"><strong>Task:</strong> for each station and day, predict whether total passengers (arrivals + departures) will be above that station’s normal (median) level. <strong>Features:</strong> station, day of week, weekend, month, year, day of month, yesterday’s passengers and the same weekday last week. <strong>Evaluation:</strong> stratified 10-fold cross-validation on {len(m):,} rows, mean ± standard deviation.</p>
<div class="m3Scroll"><table class="m3Table"><thead><tr><th>Model</th><th class="n">Precision</th><th class="n">Recall</th><th class="n">F1</th><th class="n">AUC</th></tr></thead><tbody>{res_rows}</tbody></table></div>
<p class="m3P">The baseline always predicts the most common class, so its AUC is 0.5 and it shows the score a model must beat. <strong>{e(best['model'])}</strong> is best, with F1 {best['f1']:.3f} ({gain:+.3f} over the baseline) and AUC {best['roc_auc']:.3f}. Most useful features: {e(top_txt)}.</p>

<h3 class="m3H">3. First draft: what works, what doesn’t, how we’ll fix it</h3>
<div class="m3Draft">
 <div><h4>What works</h4><ul>
  <li>A repeatable Python pipeline cleans the raw file and logs every change.</li>
  <li>{beat_txt}</li>
  <li>Strongest signals: {e(top_txt)}.</li></ul></div>
 <div><h4>What doesn’t work yet</h4><ul>
  <li>Random 10-fold CV mixes future and past days. When we train only on the past and test on the future, F1 is {ts_f1:.3f} (AUC {ts_auc:.3f}), {abs(drop):.3f} {'lower' if drop > 0 else 'higher'} than random CV.</li>
  <li>Station-level data exists only for the Airport Rail Link.</li>
  <li>Public holidays are not a feature yet, so long weekends can be misread.</li>
  <li>The map timeline does not read this real data yet.</li></ul></div>
 <div><h4>How we’ll fix it</h4><ul>
  <li>Report time-based validation next to the 10-fold results.</li>
  <li>Add a Thai public-holiday calendar as a feature.</li>
  <li>Connect the cleaned Airport Rail Link data to the station timeline.</li>
  <li>Search for station-level files for BTS and MRT; then cluster stations by demand pattern.</li></ul></div>
</div>
</section>
"""

css = """
.m3Facts{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:14px;overflow:hidden;margin:6px 0 34px}
.m3Facts div{background:#fff;padding:18px 20px}.m3Facts b{display:block;font-size:22px;letter-spacing:-.02em}.m3Facts span{color:var(--muted);font-size:13px}
.m3H{font-size:20px;letter-spacing:-.02em;margin:34px 0 8px}.m3P{color:var(--muted);max-width:820px;margin:0 0 14px}
.m3Scroll{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:#fff}
.m3Table{width:100%;border-collapse:collapse;font-size:14px;min-width:560px}.m3Table th,.m3Table td{padding:11px 14px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
.m3Table th{font-size:12px;color:var(--muted);font-weight:700;background:#f8faf9}.m3Table tr:last-child td{border-bottom:0}
.m3Table .n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.m3Table small{display:block;color:var(--muted);font-size:11px}
.m3Table tr.best td{background:#eaf8f3;font-weight:650}
.m3Draft{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.m3Draft>div{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px 20px}
.m3Draft h4{margin:0 0 8px;font-size:15px}.m3Draft ul{margin:0;padding-left:18px;color:var(--muted);font-size:14px}.m3Draft li{margin:6px 0}
@media(max-width:820px){.m3Facts{grid-template-columns:repeat(2,1fr)}.m3Draft{grid-template-columns:1fr}}
@media print{.m3{display:none}}
"""

page = HTML_IN.read_text(encoding="utf-8")
fixes = [
    ("Use the uploaded rail map below.", "Use the rail map below."),
    ("based on uploaded railway network schematic", "based on a railway network schematic"),
    ("Explore the uploaded network map.", "Explore the network map."),
    ("The map is your supplied reference SVG (schematic / not to scale).", "The map is a schematic (not to scale)."),
    ("The map is based on the supplied SVG; government-data integration across the network remains in progress.",
     "The map is a schematic. Station-level daily data is currently available for the Airport Rail Link only."),
    ('<small id="timeLabel">2024–2025</small>', '<small id="timeLabel">Jan 2022 – Dec 2024</small>'),
    ("We will inspect the actual CSV for line coverage before enabling daily station histories across the network.",
     f"Audited in Milestone 3: covers the Airport Rail Link ({n_st} stations), daily, {d0:%b %Y} – {d1:%b %Y}."),
    ("<h3>Data mining extension</h3>", "<h3>Data mining track</h3>"),
    ("Proposed secondary track: station-pattern clustering, pending data suitability and Ajarn’s approval.",
     "Milestone 3: high-demand day classification is now the core mining task (see Progress). Station clustering is the next step."),
    ('<a href="#method">Method</a>', '<a href="#method">Method</a><a href="#progress">Progress</a>'),
    ("<title>Bangkok Rail Explorer | ICT324 Milestone 2</title>", "<title>Bangkok Rail Explorer | ICT324 Milestone 3</title>"),
    ("<span> · ICT324 Milestone 2 concept</span>", "<span> · ICT324 Milestone 3 progress</span>"),
    ("05 / Planned outcomes", "06 / Planned outcomes"),
    ("06 / Project group", "07 / Project group"),
]
for old, new in fixes:
    if old in page:
        page = page.replace(old, new, 1)
    else:
        print(f"[warn] text not found, skipped: {old[:60]}")

anchor = '<section class="section next" id="deliverables">'
if anchor not in page:
    sys.exit("Could not find where to insert the Progress section.")
page = page.replace(anchor, section + anchor, 1)
page = page.replace("</style>", css + "</style>", 1)
(OUT / "index.html").write_text(page, encoding="utf-8")
print(f"\nDone. Open {OUT/'index.html'} and upload it as the team webpage.")
