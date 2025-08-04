import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --- Load Data ---
df = pd.read_excel("Data/EU_data.xlsx")

# --- Data Preparation ---

# Municipality type mapping/cleaning
municipality_type_map = {
    "Large town/ city (over 50,000 people)": "Large town/city",
    "Small or medium-sized town (50,000 people or less)": "Small/medium town",
    "A rural area or village": "Rural"
}
df = df[df['municipality_type'].isin(municipality_type_map.keys())]
df['municipality_type_group'] = df['municipality_type'].map(municipality_type_map)

# Municipality years banding (years_band)
df['municipality_years'] = pd.to_numeric(df['municipality_years'], errors='coerce')
df = df[df['municipality_years'].between(0, 120)]
years_bins = [0, 5, 15, 25, 35, 45, np.inf]
years_labels = ['0–5', '6–15', '16–25', '26–35', '36–45', '46+']
df['years_band'] = pd.cut(df['municipality_years'], bins=years_bins, labels=years_labels, right=True)
df = df[~df['municipality_type_group'].isna()]
df = df[~df['years_band'].isna()]

# Covariates
# Age banding
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df = df[df['age'].between(16, 120)]
age_bins = [16, 24, 34, 44, 54, 64, np.inf]
age_labels = ['16–24', '25–34', '35–44', '45–54', '55–64', '65+']
df['age_band'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True)

# Recode income (for covariate)
income_map = {
    'Decile 1': 'Low', 'Decile 2': 'Low', 'Decile 3': 'Low',
    'Decile 4': 'Medium', 'Decile 5': 'Medium', 'Decile 6': 'Medium', 'Decile 7': 'Medium',
    'Decile 8': 'High', 'Decile 9': 'High', 'Decile 10': 'High'
}
df['income_group'] = df['income_decile'].map(income_map)

# Recode education (for covariate)
education_map = {
    'Not completed primary': 'Primary or less',
    'Completed primary': 'Primary or less',
    'Completed secondary': 'Secondary',
    'Completed post secondary vocational studies, or higher education to bachelor level or equivalent': 'Post-secondary/Third Level',
    'Completed upper level of education to master, doctoral degree or equivalent': 'Post-secondary/Third Level'
}
df['education_group'] = df['education'].map(education_map)

# Country group & weighting
df['country_group'] = df['country'].apply(lambda x: 'Ireland' if x == 'Ireland' else 'EU')
df['weight'] = df.apply(lambda row: row['w_country'] if row['country_group'] == 'Ireland' else row['w_eu27'], axis=1)

# --- Recode Loneliness Variables ---
# Direct
df['loneliness_direct_bin'] = df['loneliness_direct'].map({
    "All of the time": 1,
    "Most of the time": 1,
    "Some of the time": 1,
    "A little of the time": 0,
    "None of the time": 0
})

# UCLA (total score)
ucla_map = {
    "Hardly ever or never": 1,
    "Some of the time": 2,
    "Often": 3
}
for col in ['loneliness_ucla_a', 'loneliness_ucla_b', 'loneliness_ucla_c']:
    df[col + '_score'] = df[col].map(ucla_map)
df['loneliness_ucla_total'] = df[[f"{c}_score" for c in ['loneliness_ucla_a', 'loneliness_ucla_b', 'loneliness_ucla_c']]].sum(axis=1)

# DJG
neg_items = ['loneliness_djg_a', 'loneliness_djg_b', 'loneliness_djg_c']
pos_items = ['loneliness_djg_d', 'loneliness_djg_e', 'loneliness_djg_f']
for item in neg_items:
    df[item + '_recoded'] = df[item].map({'Yes': 1, 'More or less': 1, 'No': 0})
for item in pos_items:
    df[item + '_recoded'] = df[item].map({'Yes': 0, 'More or less': 1, 'No': 1})
df['loneliness_djg_total'] = df[[col + '_recoded' for col in neg_items + pos_items]].sum(axis=1)

# --- Weighted Summary Statistics ---
def weighted_mean(x, val, w):
    x = x.dropna(subset=[val, w])
    return np.average(x[val], weights=x[w]) if not x.empty else np.nan

summary = df.groupby(['country_group', 'municipality_type_group', 'years_band'], observed=True).apply(
    lambda x: pd.Series({
        'n': len(x),
        'n_weighted': x['weight'].sum(),
        'loneliness_direct': weighted_mean(x, 'loneliness_direct_bin', 'weight'),
        'loneliness_ucla': weighted_mean(x, 'loneliness_ucla_total', 'weight'),
        'loneliness_djg': weighted_mean(x, 'loneliness_djg_total', 'weight'),
    })
).reset_index()
summary.to_csv("municipality_summary_weighted_means.csv", index=False)

# --- Visualization ---
def plot_weighted_bar(df, x_col, y_col, hue_col, title, filename):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df[df[y_col].notna()],
        x=x_col,
        y=y_col,
        hue=hue_col,
        estimator=lambda y: np.average(y, weights=df.loc[y.index, 'weight']),
        errorbar=None
    )
    plt.title(title)
    plt.ylabel("Weighted Mean")
    plt.xlabel(x_col.replace('_', ' ').title())
    plt.legend(title=hue_col.replace('_', ' ').title())
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# By municipality type
plot_weighted_bar(df, 'municipality_type_group', 'loneliness_direct_bin', 'country_group', "Direct Loneliness by Municipality Type", "municipality_loneliness_direct.png")
plot_weighted_bar(df, 'municipality_type_group', 'loneliness_ucla_total', 'country_group', "UCLA Score by Municipality Type", "municipality_ucla.png")
plot_weighted_bar(df, 'municipality_type_group', 'loneliness_djg_total', 'country_group', "DJG Score by Municipality Type", "municipality_djg.png")

# By years_band (residency)
plot_weighted_bar(df, 'years_band', 'loneliness_direct_bin', 'country_group', "Direct Loneliness by Municipality Years", "municipality_years_loneliness_direct.png")
plot_weighted_bar(df, 'years_band', 'loneliness_ucla_total', 'country_group', "UCLA Score by Municipality Years", "municipality_years_ucla.png")
plot_weighted_bar(df, 'years_band', 'loneliness_djg_total', 'country_group', "DJG Score by Municipality Years", "municipality_years_djg.png")

# --- Forest plot (from migration analysis) ---
def plot_forest(models, labels, filename):
    rows = []
    for model, label in zip(models, labels):
        for param in model.params.index:
            if param == 'Intercept':
                continue
            ci = model.conf_int().loc[param]
            rows.append({
                'Variable': param,
                'Estimate': model.params[param],
                'CI_lower': ci[0],
                'CI_upper': ci[1],
                'Model': label
            })
    df_plot = pd.DataFrame(rows).sort_values(by='Variable')
    plt.figure(figsize=(12, 18))
    sns.pointplot(
        data=df_plot, x='Estimate', y='Variable', hue='Model',
        join=False, dodge=0.5, capsize=0.2, errwidth=1, errorbar=None
    )
    for i, row in df_plot.iterrows():
        plt.plot([row['CI_lower'], row['CI_upper']], [i, i], color='black', lw=1)
    plt.axvline(x=0, linestyle='--', color='grey')
    plt.title("Regression Coefficients")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_residuals(model, filename, title):
    plt.figure(figsize=(8,4))
    sns.histplot(model.resid, bins=30, kde=True)
    plt.title(title)
    plt.xlabel("Residual")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- Regression: Main effect and interaction, post-hoc (no plotting here) ---
def run_municipality_model(data, label, outcome_var):
    model_df = data.dropna(subset=[
        outcome_var, 'municipality_type_group', 'years_band', 'age_band', 'income_group', 'education_group', 'weight'
    ])
    model = smf.wls(
        f"{outcome_var} ~ C(municipality_type_group) + C(years_band) + C(age_band) + C(income_group) + C(education_group)",
        data=model_df,
        weights=model_df['weight']
    ).fit()
    with open(f"municipality_{label}_{outcome_var}.txt", "w") as f:
        f.write(model.summary().as_text())
    for factor in ['municipality_type_group', 'years_band']:
        valid = model_df.dropna(subset=[outcome_var, factor])
        if valid[factor].nunique() > 1:
            try:
                posthoc = pairwise_tukeyhsd(
                    endog=valid[outcome_var],
                    groups=valid[factor],
                    alpha=0.05
                )
                with open(f"posthoc_{factor}_{label}_{outcome_var}.txt", "w") as f:
                    f.write(str(posthoc.summary()))
            except Exception as e:
                print(f"Posthoc failed for {factor}-{label}-{outcome_var}: {e}")
    return model

def run_interaction_model(data, label, outcome_var):
    model_df = data.dropna(subset=[
        outcome_var, 'municipality_type_group', 'years_band', 'age_band', 'income_group', 'education_group', 'weight'
    ])
    model = smf.wls(
        f"{outcome_var} ~ C(municipality_type_group) * C(years_band) + C(age_band) + C(income_group) + C(education_group)",
        data=model_df,
        weights=model_df['weight']
    ).fit()
    with open(f"municipality_{label}_{outcome_var}_interaction.txt", "w") as f:
        f.write(model.summary().as_text())
    return model

# --- Comparative modeling and plotting ---
for outcome in ['loneliness_direct_bin', 'loneliness_ucla_total', 'loneliness_djg_total']:
    main_models, main_labels = [], []
    inter_models, inter_labels = [], []
    for group, label in [("Ireland", "Ireland"), ("EU", "EU")]:
        data = df[df['country_group'] == group]
        main_models.append(run_municipality_model(data, label, outcome))
        main_labels.append(label)
        inter_models.append(run_interaction_model(data, label, outcome))
        inter_labels.append(label)
        plot_residuals(main_models[-1], f"municipality_{label.lower()}_{outcome}_resid.png", f"Residuals: {label}, {outcome}")
        plot_residuals(inter_models[-1], f"municipality_{label.lower()}_{outcome}_interaction_resid.png", f"Interaction Residuals: {label}, {outcome}")
    plot_forest(main_models, main_labels, f"municipality_comparative_{outcome}_coefplot.png")
    plot_forest(inter_models, inter_labels, f"municipality_comparative_{outcome}_interaction_coefplot.png")

# --- Weighted Chi-Square (Direct Loneliness) ---
def weighted_chi_square(data, group_col, outcome_col, weight_col, label):
    df_valid = data.dropna(subset=[group_col, outcome_col, weight_col])
    ct = pd.pivot_table(
        df_valid,
        index=outcome_col,
        columns=group_col,
        values=weight_col,
        aggfunc='sum'
    ).fillna(0)
    chi2, p, dof, expected = chi2_contingency(ct)
    result = f"""Chi-square Test: {label}
Chi2 statistic: {chi2:.4f}
Degrees of freedom: {dof}
p-value: {p:.4g}
Observed (weighted counts):
{ct.to_string()}
Expected counts:
{pd.DataFrame(expected, index=ct.index, columns=ct.columns).round(2).to_string()}
"""
    with open(f"chi_square_{group_col}_{label}.txt", "w") as f:
        f.write(result)

for group, label in [("Ireland", "ireland"), ("EU", "eu")]:
    data = df[df['country_group'] == group]
    weighted_chi_square(data, 'municipality_type_group', 'loneliness_direct_bin', 'weight', label)
    weighted_chi_square(data, 'years_band', 'loneliness_direct_bin', 'weight', label)
