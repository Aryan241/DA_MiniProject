# Factors Influencing Student Exam Performance

CES 2103 Data Analytics — IA3 mini project, Manipal Institute of Technology.

This project explores exam scores and tests their association with test preparation,
parental education, and lunch type. It includes the input dataset and Python code
needed to reproduce the preprocessing, exploratory data analysis (EDA), and
confirmatory data analysis (CDA).

## Team

- Sayed Sami Ahmad
- Agamvir Singh Arora
- Aryan Achary

## Dataset and attribution

`StudentsPerformance.csv` contains 1,000 rows and eight columns: gender,
race/ethnicity group, parental education, lunch type, test preparation status,
and math, reading, and writing scores.

- Source: [Students Performance in Exams, uploaded by spscientist on Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams).
- Kaggle's description credits [Royce Kimmons's generated exam data tool](http://roycekimmons.com/tools/generated_data/exams).
- Kaggle's API lists the dataset license as **Unknown** (checked 26 October 2026).
  No CC0 or other license is asserted here, and no new license is applied to the dataset.
- The original CSV used in this project is included for reproducibility. The cleaned
  CSV is derived from it by `analysis.py`.

The source points to generated data; these records should be treated as an
illustrative teaching dataset, not a verified representative sample of real students.

## Run the analysis

Use Python 3.13 and run the following from a terminal:

```bash
git clone https://github.com/Aryan241/DA_MiniProject.git
cd DA_MiniProject
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis.py
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
Run `analysis.py` from the repository directory. It prints the numerical results
and regenerates the cleaned CSV, `results.json`, and all five PNG figures,
overwriting the corresponding generated files. No notebook or Kaggle credentials
are needed. Dependency versions are recorded in `requirements.txt`.

Verified with Python 3.13: the script completed successfully, reproduced every
saved statistic within numerical tolerance, produced an identical cleaned CSV,
and generated all five figures. `pip check` reported no dependency conflicts.

## Methods

Preprocessing checks missing values, duplicate rows, and score ranges. Valid
low-score outliers are retained. Column names are standardized, categorical
variables encoded, and an average of the three scores is calculated and rounded
to two decimals. Standardized score columns are also saved; they are not used in
the hypothesis tests.

EDA includes descriptive statistics, score distributions, group boxplots, a
correlation heatmap, and parental-education means with 95% confidence intervals.

CDA uses a significance level of 0.05:

| Question | Test | Supporting analysis |
| --- | --- | --- |
| Do average scores differ by test preparation status? | Two-sided Welch t-test | Mean-difference CI, Cohen's d, Mann–Whitney U |
| Do average scores differ across parental education categories? | One-way ANOVA | Levene test, Tukey HSD, eta squared, Kruskal–Wallis |
| Is lunch type associated with performance band? | Chi-square independence test | Expected counts and Cramér's V |

Performance bands use `pandas.qcut`: Low ≤62; Medium >62 and ≤74.33;
High >74.33. Ties produce 336, 332, and 332 records respectively.

## Results

| Analysis | Result |
| --- | --- |
| Test preparation | Means 72.67 vs 65.04; difference 7.63, 95% CI [5.89, 9.37]; t = 8.595; p < 0.001; d = 0.553 |
| Parental education | F(5, 994) = 10.753; p < 0.001; eta squared = 0.0513 |
| Lunch and performance band | Chi-square(2) = 64.56; p < 0.001; Cramér's V = 0.254 |

These are associations within this dataset, not evidence that any factor causes
higher scores. Sampling and independence cannot be verified from the available
provenance. The parental-education Pearson correlation depends on the chosen
0–5 coding and its equal-spacing assumption. Correlations with the average score
partly reflect that the average contains the subject scores themselves.

## Repository files

| File | Purpose |
| --- | --- |
| `analysis.py` | Complete preprocessing, EDA, and CDA code |
| `StudentsPerformance.csv` | Original input dataset |
| `requirements.txt` | Tested Python dependencies |
| `cleaned_students_performance.csv` | Generated data with derived columns |
| `results.json` | Generated descriptive and inferential statistics |
| `fig1_dist.png` | Subject score distributions |
| `fig2_box.png` | Group comparisons |
| `fig3_corr.png` | Correlation heatmap |
| `fig4_edu.png` | Parental education means and confidence intervals |
| `fig5_chi.png` | Performance bands by lunch type |

The report is submitted separately through the LMS. Repository link for the
end of the report: https://github.com/Aryan241/DA_MiniProject
