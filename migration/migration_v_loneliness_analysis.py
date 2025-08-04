import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# --- Plotting Functions ---

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
    df_plot = pd.DataFrame(rows)
    # Only plot present parameters, sorted for compactness
    yorder = df_plot['Variable'].unique()
    df_plot['Variable'] = pd.Categorical(df_plot['Variable'], categories=yorder[::-1], ordered=True)
    plt.figure(figsize=(12, 2 + 0.5 * len(yorder)))
    if df_plot['Model'].nunique() > 1:
        sns.pointplot(
            data=df_plot, x='Estimate', y='Variable', hue='Model',
            join=False, dodge=0.5, capsize=0.2, errwidth=1, errorbar=None
        )
    else:
        sns.pointplot(
            data=df_plot, x='Estimate', y='Variable',
            join=False, capsize=0.2, errwidth=1, errorbar=None
        )
    # Only plot CIs where the row is present (no blank lines)
    for i, row in df_plot.iterrows():
        yval = list(yorder[::-1]).index(row['Variable'])
        plt.plot([row['CI_lower'], row['CI_upper']], [yval, yval], color='black', lw=1)
    plt.axvline(x=0, linestyle='--', color='grey')
    plt.title("Regression Coefficients")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_residuals(model, filename, title):
    plt.figure(figsize=(8, 4))
    sns.histplot(model.resid, bins=30, kde=True)
    plt.title(title)
    plt.xlabel("Residual")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- Load Data ---

df = pd.read_excel("Data/EU_data.xlsx")

# --- Data Preparation ---

df['migrant_flag'] = (
    (df['country_birth_yourself'] != df['country']) |
    (df['country_birth_mother'] != df['country']) |
    (df['country_birth_father'] != df['country'])
).astype(int)

df['country_group'] = df['country'].apply(lambda x: 'Ireland' if x == 'Ireland' else 'EU')
df['weight'] = df.apply(lambda row: row['w_country'] if row['country_group'] == 'Ireland' else row['w_eu27'], axis=1)

# Recode loneliness_direct
df['loneliness_direct_bin'] = df['loneliness_direct'].map({
    "All of the time": 1,
    "Most of the time": 1,
    "Some of the time": 1,
    "A little of the time": 0,
    "None of the time": 0
})

# Recode UCLA
ucla_map = {
    "Hardly ever or never": 1,
    "Some of the time": 2,
    "Often": 3
}
for col in ['loneliness_ucla_a', 'loneliness_ucla_b', 'loneliness_ucla_c']:
    df[col + '_score'] = df[col].map(ucla_map)
df['loneliness_ucla_total'] = df[[f"{c}_score" for c in ['loneliness_ucla_a', 'loneliness_ucla_b', 'loneliness_ucla_c']]].sum(axis=1)

# Recode DJG
neg_items = ['loneliness_djg_a', 'loneliness_djg_b', 'loneliness_djg_c']
pos_items = ['loneliness_djg_d', 'loneliness_djg_e', 'loneliness_djg_f']
for item in neg_items:
    df[item + '_recoded'] = df[item].map({'Yes': 1, 'More or less': 1, 'No': 0})
for item in pos_items:
    df[item + '_recoded'] = df[item].map({'Yes': 0, 'More or less': 1, 'No': 1})
df['loneliness_djg_total'] = df[[col + '_recoded' for col in neg_items + pos_items]].sum(axis=1)

# Create age_band
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df = df[df['age'].between(16, 120)]
age_bins = [16, 24, 34, 44, 54, 64, np.inf]
age_labels = ['16–24', '25–34', '35–44', '45–54', '55–64', '65+']
df['age_band'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True)

# Grouped covariates (use as in other scripts)
income_map = {
    'Decile 1': 'Low', 'Decile 2': 'Low', 'Decile 3': 'Low',
    'Decile 4': 'Medium', 'Decile 5': 'Medium', 'Decile 6': 'Medium', 'Decile 7': 'Medium',
    'Decile 8': 'High', 'Decile 9': 'High', 'Decile 10': 'High'
}
education_map = {
    'Not completed primary': 'Primary or less',
    'Completed primary': 'Primary or less',
    'Completed secondary': 'Secondary',
    'Completed post secondary vocational studies, or higher education to bachelor level or equivalent': 'Post-secondary/Third Level',
    'Completed upper level of education to master, doctoral degree or equivalent': 'Post-secondary/Third Level'
}
municipality_type_map = {
    "Large town/ city (over 50,000 people)": "Large town/city",
    "Small or medium-sized town (50,000 people or less)": "Small/medium town",
    "A rural area or village": "Rural"
}
df['income_group'] = df['income_decile'].map(income_map)
df['education_group'] = df['education'].map(education_map)
df['municipality_type_group'] = df['municipality_type'].map(municipality_type_map)

# --- Weighted Summary ---

summary = df.groupby(['country_group', 'migrant_flag'], observed=True).apply(
    lambda x: pd.Series({
        'n': len(x),
        'loneliness_direct': np.average(x['loneliness_direct_bin'].dropna(), weights=x.loc[x['loneliness_direct_bin'].notna(), 'weight']),
        'loneliness_ucla': np.average(x['loneliness_ucla_total'].dropna(), weights=x.loc[x['loneliness_ucla_total'].notna(), 'weight']),
        'loneliness_djg': np.average(x['loneliness_djg_total'].dropna(), weights=x.loc[x['loneliness_djg_total'].notna(), 'weight']),
    })
).reset_index()
summary.to_csv("migration_summary_weighted_means.csv", index=False)

# --- Visualization ---

plot_weighted_bar(df, 'migrant_flag', 'loneliness_direct_bin', 'country_group', "Loneliness Prevalence by Migration Status", "migration_loneliness_direct.png")
plot_weighted_bar(df, 'migrant_flag', 'loneliness_ucla_total', 'country_group', "UCLA Score by Migration Status", "migration_ucla_score.png")
plot_weighted_bar(df, 'migrant_flag', 'loneliness_djg_total', 'country_group', "DJG Score by Migration Status", "migration_djg_score.png")

# --- Weighted Regression, Forest Plot, and Residuals (for all outcomes) ---

def run_wls_and_plots(data, label, outcome_var):
    model_df = data.dropna(subset=[
        outcome_var, 'migrant_flag', 'age_band',
        'education_group', 'income_group', 'municipality_type_group', 'weight'
    ])
    model = smf.wls(
        f'{outcome_var} ~ migrant_flag + C(age_band) + C(education_group) + C(income_group) + C(municipality_type_group)',
        data=model_df,
        weights=model_df['weight']
    ).fit()
    with open(f"migration_logit_{label}_{outcome_var}.txt", "w") as f:
        f.write(model.summary().as_text())
    plot_residuals(model, f"migration_{label}_{outcome_var}_resid.png", f"Residuals: {label}, {outcome_var}")
    return model

for group, label in [("Ireland", "ireland"), ("EU", "eu")]:
    data = df[df['country_group'] == group]
    models, model_labels = [], []
    for outcome in ['loneliness_direct_bin', 'loneliness_ucla_total', 'loneliness_djg_total']:
        models.append(run_wls_and_plots(data, label, outcome))
        model_labels.append(f"{label}-{outcome}")
    plot_forest(models, model_labels, f"migration_{label}_forest_plot.png")

# --- Weighted Chi-Square Tests ---

def run_chi_square_test(data, label):
    df_valid = data.dropna(subset=['loneliness_direct_bin', 'migrant_flag', 'weight'])
    ct = pd.pivot_table(
        df_valid,
        index='loneliness_direct_bin',
        columns='migrant_flag',
        values='weight',
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
    with open(f"chi_square_{label}.txt", "w") as f:
        f.write(result)

for group, label in [("Ireland", "ireland"), ("EU", "eu")]:
    data = df[df['country_group'] == group]
    run_chi_square_test(data, label)
