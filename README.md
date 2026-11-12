# Bangkok Rail Explorer — Milestone 3 (The Teletubbies)

**Live GitHub Pages website:** https://iluvabees.github.io/The-Teletubbies-Bangkok-Rail-Explorer/

## How to publish the update

1. **Extract this ZIP.** Do not upload the ZIP itself to GitHub: GitHub Pages does not unpack ZIP files automatically.
2. In your existing GitHub repository, open the **main** branch and select **Add file → Upload files**.
3. Upload **all eight extracted files together at the repository root** (next to the existing `index.html`, NOT inside another folder). Allow the new `index.html` to replace the existing one.
4. Click **Commit changes**. In Settings → Pages, keep source **Deploy from a branch**, `main`, `/(root)`. The public URL stays the same.
5. Once deployed, open the site with a fresh tab and test the **Progress** section, **Print 1-page proposal**, and project download links.

**You do not need to delete the existing repository or delete all its files.** GitHub replaces only files with matching paths; unrelated existing files can remain. If there are outdated duplicate website versions, you can remove those *later*, but make sure you keep root `index.html`.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | **The live website**: responsive Bangkok Rail Explorer, The Teletubbies branding, member IDs, map, and Milestone 3 progress/results section |
| `drt2566_02.csv` | **Original raw DRT dataset** (this newly supplied file; 12,408 rows, eight Airport Rail Link stations) |
| `cleaned_data.csv` | Friend's processed station-day dataset (12,296 rows) |
| `cleaning_log.csv` | Record of cleaning steps and affected counts |
| `model_results.csv` | Supplied model comparison metrics |
| `build_milestone3.py` | Friend's original Python data-cleaning/modeling/site-building script, unmodified |
| `Bangkok_Rail_Explorer_Teletubbies.html` | Milestone 2 HTML input the builder expects (with team branding and IDs) |
| `README.md` | This publishing and reproducibility guide |

## Research and reproducibility notes

- The uploaded raw CSV was checked for its schema and counts. It has **12,408 rows**, station codes **A1–A8**, dates **1 Jan 2022–31 Mar 2026**, and **112 station-days where both counts are zero**. I reran the original cleaning portion of `build_milestone3.py` against the supplied raw file; the resulting **12,296-row, 10-column cleaned dataset matched `cleaned_data.csv` exactly**. This verifies reproducibility of the data-cleaning output, **not** the correctness of the script's cleaning assumptions or the model scores.
- The builder expects `drt2566_02.csv` and `Bangkok_Rail_Explorer_Teletubbies.html` in the same directory. Both are now included. To reproduce the computational workflow, install `numpy`, `pandas`, and `scikit-learn` and run `python build_milestone3.py`. It writes results to the `m3_output/` directory. The published `index.html` is the **reviewed and annotated** Milestone 3 website, and can differ in branding, file links, and explanatory notes from the builder's raw HTML output.
- Source: [Thailand Open Government Data — DRT dataset](https://data.go.th/th/dataset/drt2566_02). **This supplied extract** covers the Airport Rail Link; that does not prove all DRT datasets contain only Airport Rail Link data.
- The original script assumes 112 zero-count records mean no report and heuristically shifts 16 duplicate-date records to the preceding date. Confirm those assumptions against source documentation before making final research claims.
- Model results from the supplied script/site are **preliminary**. Shuffled 10-fold cross-validation gives substantially higher performance than the past-to-future test (reported Random Forest F1 about **0.923** vs **0.734**). Do not describe the shuffled-CV value as validated future forecasting accuracy.
- The Milestone 3 station-day CSV is **not yet wired into the map timeline**. The interactive map remains a demonstration. Siam BTS 2023 official monthly values are separate; other station/date values in the map remain labeled as simulated.
- The public site lists **Kyaw Zaw Lin (6702599)**, **Aung Khant Tun (6702977)**, **Aung Kaung Sett Naing (6703015)**.
