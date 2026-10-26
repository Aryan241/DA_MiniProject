"""
IA3 Mini Project - CES 2103 Data Analytics
Topic: Factors affecting student exam performance
Dataset: Students Performance in Exams (Kaggle), 1000 records
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", font_scale=0.9)
OUT = {}

# ------------------------------------------------------------------ 1. LOAD
df = pd.read_csv("StudentsPerformance.csv")
OUT["raw_shape"] = list(df.shape)

# ------------------------------------------------------------ 2. PREPROCESS
df.columns = (df.columns.str.strip().str.lower()
              .str.replace("/", "_").str.replace(" ", "_"))
df = df.rename(columns={"race_ethnicity": "group",
                        "parental_level_of_education": "parent_edu",
                        "test_preparation_course": "test_prep",
                        "math_score": "math", "reading_score": "reading",
                        "writing_score": "writing"})
scores = ["math", "reading", "writing"]

OUT["missing_total"] = int(df.isna().sum().sum())
OUT["duplicates"] = int(df.duplicated().sum())
OUT["range_violations"] = int(((df[scores] < 0) | (df[scores] > 100)).sum().sum())

# Outlier detection (IQR rule) - detected and inspected, but retained
outl = {}
for c in scores:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outl[c] = {"n": int(((df[c] < lo) | (df[c] > hi)).sum()),
               "lo": round(lo, 1), "hi": round(hi, 1)}
OUT["outliers"] = outl

# Feature engineering
df["avg_score"] = df[scores].mean(axis=1).round(2)
edu_order = ["some high school", "high school", "some college",
             "associate's degree", "bachelor's degree", "master's degree"]
df["parent_edu"] = pd.Categorical(df["parent_edu"], edu_order, ordered=True)
df["edu_code"] = df["parent_edu"].cat.codes
df["gender_code"] = (df["gender"] == "male").astype(int)
df["lunch_code"] = (df["lunch"] == "standard").astype(int)
df["prep_code"] = (df["test_prep"] == "completed").astype(int)

# Equal-frequency binning (discretisation) of average score into 3 bands
df["perf_band"] = pd.qcut(df["avg_score"], 3, labels=["Low", "Medium", "High"])
OUT["band_cuts"] = [round(x, 2) for x in
                    pd.qcut(df["avg_score"], 3, retbins=True)[1]]

# Z-score normalisation (used for comparing scales / PCA-style work)
for c in scores:
    df[c + "_z"] = ((df[c] - df[c].mean()) / df[c].std()).round(4)

df.to_csv("cleaned_students_performance.csv", index=False)

# ------------------------------------------------------------------ 3. EDA
desc = df[scores + ["avg_score"]].describe().T
desc["skew"] = df[scores + ["avg_score"]].skew()
OUT["describe"] = desc.round(2).to_dict("index")

OUT["group_means"] = {
    "gender": df.groupby("gender")["avg_score"].mean().round(2).to_dict(),
    "lunch": df.groupby("lunch")["avg_score"].mean().round(2).to_dict(),
    "test_prep": df.groupby("test_prep")["avg_score"].mean().round(2).to_dict(),
    "parent_edu": df.groupby("parent_edu", observed=True)["avg_score"]
                    .mean().round(2).to_dict(),
}
OUT["group_sizes_edu"] = df["parent_edu"].value_counts().reindex(edu_order).to_dict()
corr = df[scores + ["avg_score", "edu_code", "lunch_code", "prep_code",
                    "gender_code"]].corr().round(3)
OUT["corr_math_reading"] = float(corr.loc["math", "reading"])
OUT["corr_reading_writing"] = float(corr.loc["reading", "writing"])
OUT["corr_math_writing"] = float(corr.loc["math", "writing"])
OUT["corr_avg_lunch"] = float(corr.loc["avg_score", "lunch_code"])
OUT["corr_avg_prep"] = float(corr.loc["avg_score", "prep_code"])
OUT["corr_avg_edu"] = float(corr.loc["avg_score", "edu_code"])
OUT["corr_avg_gender"] = float(corr.loc["avg_score", "gender_code"])

# Fig 1: distributions
fig, ax = plt.subplots(1, 3, figsize=(9, 2.6), sharey=True)
for a, c, col in zip(ax, scores, ["#4C72B0", "#55A868", "#C44E52"]):
    sns.histplot(df[c], bins=20, kde=True, ax=a, color=col)
    a.set_title(f"{c.title()} score"); a.set_xlabel("Score")
ax[0].set_ylabel("Students")
plt.tight_layout(); plt.savefig("fig1_dist.png", dpi=200); plt.close()

# Fig 2: box plots by categorical factors
fig, ax = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
for a, c, t in zip(ax, ["test_prep", "lunch", "gender"],
                   ["Test preparation", "Lunch type", "Gender"]):
    sns.boxplot(data=df, x=c, y="avg_score", ax=a, palette="Set2", hue=c,
                legend=False)
    a.set_title(t); a.set_xlabel("")
ax[0].set_ylabel("Average score")
plt.tight_layout(); plt.savefig("fig2_box.png", dpi=200); plt.close()

# Fig 3: correlation heatmap
fig, ax = plt.subplots(figsize=(4.6, 3.6))
lab = {"math": "Math", "reading": "Reading", "writing": "Writing",
       "avg_score": "Average", "edu_code": "Parent edu.",
       "lunch_code": "Std. lunch", "prep_code": "Test prep",
       "gender_code": "Male"}
sns.heatmap(corr.rename(index=lab, columns=lab), annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
            annot_kws={"size": 7}, cbar_kws={"shrink": .8})
plt.tight_layout(); plt.savefig("fig3_corr.png", dpi=200); plt.close()

# Fig 4: parental education means with 95% CI
g = df.groupby("parent_edu", observed=True)["avg_score"].agg(["mean", "sem"])
g["ci"] = g["sem"] * stats.t.ppf(.975, df.groupby("parent_edu", observed=True)["avg_score"].count() - 1)
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.barh(range(len(g)), g["mean"], xerr=g["ci"], color="#4C72B0", capsize=3)
ax.set_yticks(range(len(g))); ax.set_yticklabels([s[0].upper()+s[1:] for s in g.index], fontsize=8)
ax.set_xlim(50, 80); ax.set_xlabel("Mean average score (95% CI)")
plt.tight_layout(); plt.savefig("fig4_edu.png", dpi=200); plt.close()

# ------------------------------------------------------------------ 4. CDA
# ---- Test 1: Welch two-sample t-test: avg_score by test prep course
a = df.loc[df.test_prep == "completed", "avg_score"]
b = df.loc[df.test_prep == "none", "avg_score"]
lev = stats.levene(a, b)
t = stats.ttest_ind(a, b, equal_var=False)
diff = a.mean() - b.mean()
se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
dfw = se**4 / ((a.var(ddof=1) / len(a))**2 / (len(a) - 1) +
               (b.var(ddof=1) / len(b))**2 / (len(b) - 1))
ci = (diff - stats.t.ppf(.975, dfw) * se, diff + stats.t.ppf(.975, dfw) * se)
sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) /
             (len(a) + len(b) - 2))
OUT["ttest"] = {
    "n_completed": len(a), "n_none": len(b),
    "mean_completed": round(a.mean(), 2), "mean_none": round(b.mean(), 2),
    "sd_completed": round(a.std(), 2), "sd_none": round(b.std(), 2),
    "levene_p": float(lev.pvalue), "t": round(float(t.statistic), 3),
    "df": round(float(dfw), 1), "p": float(t.pvalue),
    "diff": round(diff, 2), "ci": [round(ci[0], 2), round(ci[1], 2)],
    "cohens_d": round(diff / sp, 3),
    "mannwhitney_p": float(stats.mannwhitneyu(a, b).pvalue),
    "skew_a": round(float(stats.skew(a)), 2), "skew_b": round(float(stats.skew(b)), 2)}

# ---- Test 2: One-way ANOVA: avg_score across parental education levels
groups = [d["avg_score"].values for _, d in df.groupby("parent_edu", observed=True)]
f = stats.f_oneway(*groups)
levA = stats.levene(*groups)
grand = df["avg_score"].mean()
ssb = sum(len(x) * (x.mean() - grand) ** 2 for x in groups)
sst = ((df["avg_score"] - grand) ** 2).sum()
tk = stats.tukey_hsd(*groups)
names = edu_order
sig = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        if tk.pvalue[i, j] < 0.05:
            sig.append([names[i], names[j], round(float(tk.statistic[j, i] * -1), 2)
                        if False else round(float(groups[j].mean() - groups[i].mean()), 2),
                        float(tk.pvalue[i, j])])
OUT["anova"] = {"F": round(float(f.statistic), 3), "p": float(f.pvalue),
                "df_between": len(groups) - 1, "df_within": len(df) - len(groups),
                "eta_sq": round(float(ssb / sst), 4), "levene_p": float(levA.pvalue),
                "kruskal_p": float(stats.kruskal(*groups).pvalue),
                "tukey_sig_pairs": sig, "n_sig_pairs": len(sig)}

# ---- Test 3: Chi-square test of independence: lunch type vs performance band
ct = pd.crosstab(df["lunch"], df["perf_band"])
chi2, p, dof, exp = stats.chi2_contingency(ct)
cramers = np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1)))
OUT["chi2"] = {"table": ct.to_dict("index"), "chi2": round(float(chi2), 2),
               "dof": int(dof), "p": float(p), "cramers_v": round(float(cramers), 3),
               "min_expected": round(float(exp.min()), 1)}

# Fig 5: stacked bar for chi-square (lunch vs band)
pct = ct.div(ct.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(4.6, 2.6))
pct.plot(kind="barh", stacked=True, ax=ax,
         color=["#C44E52", "#DD8452", "#55A868"])
ax.set_xlabel("% of students"); ax.set_ylabel("")
ax.legend(title="Performance band", fontsize=7, title_fontsize=8, loc="upper center", bbox_to_anchor=(0.5,-0.28), ncol=3, frameon=False)
plt.tight_layout(); plt.savefig("fig5_chi.png", dpi=200); plt.close()

json.dump(OUT, open("results.json", "w"), indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str))
