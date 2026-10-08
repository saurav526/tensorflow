# this script simulates an A/B test, performs statistical analysis, and visualizes the results.
# while each section is commented for clarity, the script is designed to be run as a whole to generate a complete A/B testing report.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm, ttest_ind


# ============================================================
# 1. CONFIGURATION
# ============================================================

np.random.seed(42)

N_USERS = 20000

CONTROL_CONVERSION = 0.10
TREATMENT_CONVERSION = 0.115

CONTROL_AOV = 50
TREATMENT_AOV = 53

ALPHA = 0.05

os.makedirs("results", exist_ok=True)


# ============================================================
# 2. GENERATE A/B TEST DATA
# ============================================================

users = np.arange(1, N_USERS + 1)

groups = np.random.choice(
    ["Control", "Treatment"],
    size=N_USERS,
    p=[0.5, 0.5]
)

data = pd.DataFrame({
    "user_id": users,
    "group": groups
})


# ============================================================
# 3. SIMULATE CONVERSIONS
# ============================================================

data["conversion_probability"] = np.where(
    data["group"] == "Control",
    CONTROL_CONVERSION,
    TREATMENT_CONVERSION
)

data["converted"] = np.random.binomial(
    1,
    data["conversion_probability"]
)


# ============================================================
# 4. SIMULATE REVENUE
# ============================================================

def generate_revenue(row):

    if row["converted"] == 0:
        return 0

    if row["group"] == "Control":
        return np.random.normal(
            CONTROL_AOV,
            15
        )

    return np.random.normal(
        TREATMENT_AOV,
        15
    )


data["revenue"] = data.apply(
    generate_revenue,
    axis=1
)

data["revenue"] = data["revenue"].clip(lower=0)


# Remove unnecessary column

data.drop(
    columns=["conversion_probability"],
    inplace=True
)


# ============================================================
# 5. BASIC DATA EXPLORATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nMissing Values:")
print(data.isnull().sum())


# ============================================================
# 6. GROUP SUMMARY
# ============================================================

summary = data.groupby("group").agg(
    users=("user_id", "count"),
    conversions=("converted", "sum"),
    total_revenue=("revenue", "sum"),
    average_revenue=("revenue", "mean")
)

summary["conversion_rate"] = (
    summary["conversions"] /
    summary["users"]
)

summary["revenue_per_user"] = (
    summary["total_revenue"] /
    summary["users"]
)

print("\n" + "=" * 60)
print("GROUP SUMMARY")
print("=" * 60)

print(summary)


# ============================================================
# 7. CONVERSION RATES
# ============================================================

control = data[data["group"] == "Control"]
treatment = data[data["group"] == "Treatment"]

control_users = len(control)
treatment_users = len(treatment)

control_conversions = control["converted"].sum()
treatment_conversions = treatment["converted"].sum()

control_rate = (
    control_conversions /
    control_users
)

treatment_rate = (
    treatment_conversions /
    treatment_users
)

print("\n" + "=" * 60)
print("CONVERSION RATES")
print("=" * 60)

print(
    f"Control Conversion Rate: "
    f"{control_rate:.4%}"
)

print(
    f"Treatment Conversion Rate: "
    f"{treatment_rate:.4%}"
)


# ============================================================
# 8. ABSOLUTE AND RELATIVE LIFT
# ============================================================

absolute_lift = (
    treatment_rate -
    control_rate
)

relative_lift = (
    absolute_lift /
    control_rate
)

print("\n" + "=" * 60)
print("LIFT")
print("=" * 60)

print(
    f"Absolute Lift: "
    f"{absolute_lift:.4%}"
)

print(
    f"Relative Lift: "
    f"{relative_lift:.2%}"
)


# ============================================================
# 9. TWO-PROPORTION Z-TEST
# ============================================================

p1 = control_rate
p2 = treatment_rate

x1 = control_conversions
x2 = treatment_conversions

n1 = control_users
n2 = treatment_users


# Pooled conversion rate

pooled_probability = (
    x1 + x2
) / (
    n1 + n2
)


# Standard error

standard_error = np.sqrt(
    pooled_probability *
    (1 - pooled_probability) *
    (
        1 / n1 +
        1 / n2
    )
)


# Z-statistic

z_score = (
    p2 - p1
) / standard_error


# Two-tailed p-value

p_value = 2 * (
    1 - norm.cdf(abs(z_score))
)


print("\n" + "=" * 60)
print("TWO-PROPORTION Z-TEST")
print("=" * 60)

print(
    f"Z-score: {z_score:.4f}"
)

print(
    f"P-value: {p_value:.6f}"
)


if p_value < ALPHA:

    print(
        "Result: Statistically Significant"
    )

else:

    print(
        "Result: Not Statistically Significant"
    )


# ============================================================
# 10. CONFIDENCE INTERVAL FOR DIFFERENCE
# ============================================================

difference = treatment_rate - control_rate

se_difference = np.sqrt(
    (
        p1 * (1 - p1) / n1
    ) +
    (
        p2 * (1 - p2) / n2
    )
)

z_critical = norm.ppf(
    1 - ALPHA / 2
)

margin_of_error = (
    z_critical *
    se_difference
)

ci_lower = (
    difference -
    margin_of_error
)

ci_upper = (
    difference +
    margin_of_error
)

print("\n" + "=" * 60)
print("95% CONFIDENCE INTERVAL")
print("=" * 60)

print(
    f"Difference: {difference:.4%}"
)

print(
    f"Lower Bound: {ci_lower:.4%}"
)

print(
    f"Upper Bound: {ci_upper:.4%}"
)


# ============================================================
# 11. AVERAGE ORDER VALUE
# ============================================================

control_converted = control[
    control["converted"] == 1
]

treatment_converted = treatment[
    treatment["converted"] == 1
]

control_aov = control_converted[
    "revenue"
].mean()

treatment_aov = treatment_converted[
    "revenue"
].mean()

print("\n" + "=" * 60)
print("AVERAGE ORDER VALUE")
print("=" * 60)

print(
    f"Control AOV: "
    f"${control_aov:.2f}"
)

print(
    f"Treatment AOV: "
    f"${treatment_aov:.2f}"
)


# ============================================================
# 12. T-TEST FOR REVENUE
# ============================================================

control_revenue = control[
    "revenue"
]

treatment_revenue = treatment[
    "revenue"
]

t_stat, revenue_p_value = ttest_ind(
    treatment_revenue,
    control_revenue,
    equal_var=False
)

print("\n" + "=" * 60)
print("REVENUE T-TEST")
print("=" * 60)

print(
    f"T-statistic: {t_stat:.4f}"
)

print(
    f"P-value: {revenue_p_value:.6f}"
)

if revenue_p_value < ALPHA:

    print(
        "Revenue difference is statistically significant."
    )

else:

    print(
        "Revenue difference is not statistically significant."
    )


# ============================================================
# 13. EFFECT SIZE
# ============================================================

pooled_std = np.sqrt(
    (
        (
            len(control_revenue) - 1
        ) * control_revenue.std() ** 2
        +
        (
            len(treatment_revenue) - 1
        ) * treatment_revenue.std() ** 2
    )
    /
    (
        len(control_revenue)
        +
        len(treatment_revenue)
        - 2
    )
)

cohens_d = (
    treatment_revenue.mean()
    -
    control_revenue.mean()
) / pooled_std

print("\n" + "=" * 60)
print("EFFECT SIZE")
print("=" * 60)

print(
    f"Cohen's d: {cohens_d:.4f}"
)


# ============================================================
# 14. VISUALIZATION - CONVERSION RATE
# ============================================================

conversion_plot = pd.DataFrame({
    "Group": [
        "Control",
        "Treatment"
    ],
    "Conversion Rate": [
        control_rate,
        treatment_rate
    ]
})

plt.figure(figsize=(8, 5))

sns.barplot(
    data=conversion_plot,
    x="Group",
    y="Conversion Rate"
)

plt.title(
    "A/B Test - Conversion Rate"
)

plt.ylabel(
    "Conversion Rate"
)

plt.xlabel(
    "Experiment Group"
)

plt.ylim(
    0,
    max(
        conversion_plot["Conversion Rate"]
    ) * 1.3
)

for i, value in enumerate(
    conversion_plot["Conversion Rate"]
):

    plt.text(
        i,
        value,
        f"{value:.2%}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    "results/conversion_rate.png",
    dpi=300
)

plt.show()


# ============================================================
# 15. VISUALIZATION - REVENUE PER USER
# ============================================================

revenue_plot = pd.DataFrame({
    "Group": [
        "Control",
        "Treatment"
    ],
    "Revenue per User": [
        control["revenue"].mean(),
        treatment["revenue"].mean()
    ]
})

plt.figure(figsize=(8, 5))

sns.barplot(
    data=revenue_plot,
    x="Group",
    y="Revenue per User"
)

plt.title(
    "A/B Test - Revenue per User"
)

plt.ylabel(
    "Revenue ($)"
)

plt.xlabel(
    "Experiment Group"
)

plt.tight_layout()

plt.savefig(
    "results/revenue_per_user.png",
    dpi=300
)

plt.show()


# ============================================================
# 16. REVENUE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=data,
    x="group",
    y="revenue"
)

plt.title(
    "Revenue Distribution by Group"
)

plt.xlabel(
    "Experiment Group"
)

plt.ylabel(
    "Revenue ($)"
)

plt.tight_layout()

plt.savefig(
    "results/revenue_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 17. FINAL BUSINESS DECISION
# ============================================================

print("\n" + "=" * 60)
print("FINAL BUSINESS DECISION")
print("=" * 60)

if (
    p_value < ALPHA
    and treatment_rate > control_rate
):

    print(
        "RECOMMENDATION: Launch the Treatment version."
    )

    print(
        "The treatment produced a statistically "
        "significant improvement in conversion rate."
    )

else:

    print(
        "RECOMMENDATION: Do not launch the Treatment version yet."
    )

    print(
        "The experiment does not provide sufficient "
        "statistical evidence of improvement."
    )


# ============================================================
# 18. SAVE DATA
# ============================================================

data.to_csv(
    "results/ab_test_data.csv",
    index=False
)

summary.to_csv(
    "results/ab_test_summary.csv"
)


# ============================================================
# 19. SAVE EXPERIMENT RESULTS
# ============================================================

results = {
    "Control Users": control_users,
    "Treatment Users": treatment_users,
    "Control Conversions": control_conversions,
    "Treatment Conversions": treatment_conversions,
    "Control Conversion Rate": control_rate,
    "Treatment Conversion Rate": treatment_rate,
    "Absolute Lift": absolute_lift,
    "Relative Lift": relative_lift,
    "Z Score": z_score,
    "P Value": p_value,
    "CI Lower": ci_lower,
    "CI Upper": ci_upper,
    "Control AOV": control_aov,
    "Treatment AOV": treatment_aov,
    "Revenue T Statistic": t_stat,
    "Revenue P Value": revenue_p_value,
    "Cohens D": cohens_d
}

results_df = pd.DataFrame(
    [results]
)

results_df.to_csv(
    "results/experiment_results.csv",
    index=False
)


print("\nResults saved in the 'results' folder.")

print("\nExperiment completed successfully.")