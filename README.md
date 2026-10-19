# Sports-centre membership analysis

BU7141 Group 1. Customer demographics, attendance, registrations, membership
endings and retention. The main analysis covers 2015–2018, with follow-up through
31 October 2019.

## Run

Use Python 3.10 or newer. From the extracted project folder:

```bash
python -m pip install -r requirements.txt
python run.py
python -m unittest discover -s tests -v
```

The `results/` folder already contains the calculated tables and SQLite database.
Running the script rebuilds them from `data/raw.xlsx`.

## Follow the analysis

Open these notebooks in order in Jupyter or VS Code. Each includes its results.
For JupyterLab, install it with `python -m pip install jupyterlab` and start it
with `python -m jupyterlab`.

| Notebook | Content |
|---|---|
| [00 — Data preparation](notebooks/00_Data_Preparation.ipynb) | Source checks, selected fields, date calculations and database. |
| [01 — Monthly memberships](notebooks/01_Monthly_Membership_Analysis.ipynb) | Monthly counts, annual totals and seasonality. |
| [02 — Age groups](notebooks/02_Age_Group_Analysis.ipynb) | Estimated entry ages 18–24 and 35+, registration shares and dropout rates. |
| [03 — Retention](notebooks/03_Retention_Analysis.ipynb) | 90-/180-day outcomes and comparisons adjusted for joining month and year. |

Calculations are in [src/analysis.py](src/analysis.py). The four-entity database
model is in [database/schema.sql](database/schema.sql).

## Use the results

CSV tables are grouped under `results/data`, `monthly`, `age_groups`, `retention`
and `descriptive`. Import them into Tableau or open them in a spreadsheet.

For the main age comparison, use `MeanAnnualEnrolmentShareA/B` and
`MeanDropoutRateA/B` in `age_groups/SeasonalComparison.csv`. For adjusted
retention, use `retention/Summary.csv`, select `Scope = All 2015-2018`, and plot
`MatchedRateA/B` separately for each `HorizonDays`. A is 18–24; B is 35+.
Rates are stored as fractions: format `0.25` as 25%.

Activity, attendance and renewal summaries use the full source history and
recorded age. Membership counts reconstruct the available first-to-final
customer intervals; retention measures recorded final exits, rather than payment
continuity. These results do not provide a revenue forecast.

## Sources and contributors

Pinheiro, P., & Cavique, L. (2020). *Sport Services Dataset* (Version 1).
Mendeley Data. https://doi.org/10.17632/yprk4jdgnv.1
Dataset licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.

Field definitions: https://doi.org/10.1016/j.dib.2021.107054.

Monthly, age-group and retention analyses: Yinran Wang.
Repository preparation and database implementation: Dmytro Konovalov, BU7141 Group 1.
