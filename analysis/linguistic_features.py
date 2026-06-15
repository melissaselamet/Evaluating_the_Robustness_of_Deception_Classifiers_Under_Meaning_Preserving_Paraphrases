import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm


theory_features = [
    "ppron", "i", "we", "socrefs",
    "Authentic", "WPS", "focuspast", "visual", "auditory", "feeling",
    "Analytic", "WC", "BigWords",
    "number", "motion", "space", "time",
    "Tone", "tone_pos", "tone_neg",
    "cogproc", "insight", "cause",
]

## Load LIWC outputs for each condition
orig_liwc   = pd.read_csv("../data/liwc_original.csv")[theory_features]
ns_8b_liwc  = pd.read_csv("../data/liwc_ns_8b.csv")[theory_features]
s_8b_liwc   = pd.read_csv("../data/liwc_s_8b.csv")[theory_features]
ns_70b_liwc = pd.read_csv("../data/liwc_ns_70b.csv")[theory_features]
s_70b_liwc  = pd.read_csv("../data/liwc_s_70b.csv")[theory_features]

## Load results for flip indicators and metadata
df = pd.read_csv("../data/results.csv")
df["flip_ns_8b"]  = (df["pred_original"] != df["pred_non_strategic_8b"]).astype(int)
df["flip_s_8b"]   = (df["pred_original"] != df["pred_strategic_8b"]).astype(int)
df["flip_ns_70b"] = (df["pred_original"] != df["pred_non_strategic_70b"]).astype(int)
df["flip_s_70b"]  = (df["pred_original"] != df["pred_strategic_70b"]).astype(int)

## Compute feature differences (paraphrase - original)
# fillna(0) handles cases where LIWC could not compute a score (e.g. very short texts)
ns_8b_diff  = (ns_8b_liwc  - orig_liwc).fillna(0)
s_8b_diff   = (s_8b_liwc   - orig_liwc).fillna(0)
ns_70b_diff = (ns_70b_liwc - orig_liwc).fillna(0)
s_70b_diff  = (s_70b_liwc  - orig_liwc).fillna(0)

ns_8b_diff.columns  = [f"ns8b_diff_{c}"  for c in ns_8b_diff.columns]
s_8b_diff.columns   = [f"s8b_diff_{c}"   for c in s_8b_diff.columns]
ns_70b_diff.columns = [f"ns70b_diff_{c}" for c in ns_70b_diff.columns]
s_70b_diff.columns  = [f"s70b_diff_{c}"  for c in s_70b_diff.columns]

## Print mean feature differences split by true label
for label_val, label_name in [(0, "truthful"), (1, "deceptive")]:
    idx = df["label"] == label_val
    print(f"\n=== mean feature differences for {label_name} reviews ===")

    print(f"\n  8B non-strategic vs original:")
    for col in ns_8b_diff.columns:
        print(f"    {col}: {ns_8b_diff[idx][col].mean():.4f}")

    print(f"\n  8B strategic vs original:")
    for col in s_8b_diff.columns:
        print(f"    {col}: {s_8b_diff[idx][col].mean():.4f}")

    print(f"\n  70B non-strategic vs original:")
    for col in ns_70b_diff.columns:
        print(f"    {col}: {ns_70b_diff[idx][col].mean():.4f}")

    print(f"\n  70B strategic vs original:")
    for col in s_70b_diff.columns:
        print(f"    {col}: {s_70b_diff[idx][col].mean():.4f}")

## Logistic regression for each condition

def report_logistic_regression(X, y, feature_names, condition_name):
    """Fit logistic regression and report results in APA 7 format.
    B = unstandardized coefficient, SE = standard error,
    Wald = (B/SE)^2, p = two-tailed p-value, OR = odds ratio (exp(B))."""
    lr = LogisticRegression(max_iter=1000, random_state=11)
    lr.fit(X, y)

    # Compute standard errors via weighted covariance matrix
    p_hat = lr.predict_proba(X)[:, 1]
    W = p_hat * (1 - p_hat)
    X_weighted = X * W[:, np.newaxis]
    cov_matrix = np.linalg.pinv(X_weighted.T @ X)
    se = np.sqrt(np.diag(cov_matrix))

    coef      = lr.coef_[0]
    wald      = (coef / se) ** 2
    p_values  = (1 - norm.cdf(np.abs(coef / se))) * 2
    or_values = np.exp(coef)

    results = pd.DataFrame({
        "feature": feature_names,
        "B":    coef,
        "SE":   se,
        "Wald": wald,
        "p":    p_values,
        "OR":   or_values,
    }).sort_values("B", key=abs, ascending=False).head(5)

    print(f"\n  {condition_name}:")
    print(f"  {'Feature':<25} {'B':>7} {'SE':>7} {'Wald':>8} {'p':>7} {'OR':>7}")
    for _, row in results.iterrows():
        print(f"  {row['feature']:<25} {row['B']:>7.3f} {row['SE']:>7.3f} "
              f"{row['Wald']:>8.3f} {row['p']:>7.3f} {row['OR']:>7.3f}")

print("\n\nlogistic regression results:")
print("outcome variable: label flip (1 = flip, 0 = no flip)")
print("predictors: standardized LIWC-22 feature differences (paraphrase minus original)")

report_logistic_regression(
    StandardScaler().fit_transform(ns_8b_diff), df["flip_ns_8b"],
    ns_8b_diff.columns.tolist(), "8B non-strategic"
)
report_logistic_regression(
    StandardScaler().fit_transform(s_8b_diff), df["flip_s_8b"],
    s_8b_diff.columns.tolist(), "8B strategic"
)
report_logistic_regression(
    StandardScaler().fit_transform(ns_70b_diff), df["flip_ns_70b"],
    ns_70b_diff.columns.tolist(), "70B non-strategic"
)
report_logistic_regression(
    StandardScaler().fit_transform(s_70b_diff), df["flip_s_70b"],
    s_70b_diff.columns.tolist(), "70B strategic"
)

## Save feature differences
features_df = pd.concat([
    df[["label", "polarity", "flip_ns_8b", "flip_s_8b", "flip_ns_70b", "flip_s_70b"]],
    ns_8b_diff, s_8b_diff, ns_70b_diff, s_70b_diff
], axis=1)
features_df.to_csv("../data/linguistic_features.csv", index=False)