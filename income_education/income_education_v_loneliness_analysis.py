import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data ---
df = pd.read_excel("Data/EU_data.xlsx")

# --- Data Preparation ---

df = df[df['income_decile'].str.startswith('Decile')]
df = df[df['education'] != 'Prefer not to say']

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

df['age'] = pd.to_numeric(df['age'], errors='coerce')
df = df[df['age'].between(16, 120)]
age_bins = [16, 24, 34, 44, 54, 64, np.inf]
age_labels = ['16–24', '25–34', '35–44', '45–54', '55–64', '65+']
df['age_band'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True)

df['country_group'] = df['country'].apply(lambda x: 'Ireland' if x == 'Ireland' else 'EU')
df['weight'] = df.apply(lambda row: row['w_country'] if row['country_group'] == 'Ireland' else row['w_eu27'], axis=1)

# --- Loneliness Variables ---

df['loneliness_direct_bin'] = df['loneliness_direct'].map({
    "All of the time": 1,
    "Most of the time": 1,
    "Some of the time": 1,
    "A little of the time": 0,
    "None of the time": 0
})

ucla_map = {
    "Hardly ever or never": 1,
    "Some of the time": 2,
    "Often": 3
}
for col in ['loneliness_ucla_a', 'loneliness_ucla_b', 'loneliness_ucla_c']:
    df[col + '_score'] = df[col].map(ucla_map)
df['loneliness_ucla_total'] = df[[f"{c}_score" for c in ['loneliness_ucla_a', 'loneliness_ucla_b', 'loneliness_ucla_c']]].sum(axis=1)

neg_items = ['loneliness_djg_a', 'loneliness_djg_b', 'loneliness_djg_c']
pos_items = ['loneliness_djg_d', 'loneliness_djg_e', 'loneliness_djg_f']
for item in neg_items:
    df[item + '_recoded'] = df[item].map({'Yes': 1, 'More or less': 1, 'No': 0})
for item in pos_items:
    df[item + '_recoded'] = df[item].map({'Yes': 0, 'More or less': 1, 'No': 1})
df['loneliness_djg_total'] = df[[col + '_recoded' for col in neg_items + pos_items]].sum(axis=1)

# --- Helper: Missingness Table ---
def compute_missingness(df, group_cols, outcomes, output="missingness_table.csv"):
    results = []
    for name, group in df.groupby(group_cols):
        n_total = len(group)
        row = dict(zip(group_cols if isinstance(name, tuple) else [group_cols], name if isinstance(name, tuple) else [name]))
        row['n'] = n_total
        for outcome in outcomes:
            n_missing = group[outcome].isna().sum()
            row[f"{outcome}_missing_%"] = round(100 * n_missing / n_total, 1)
        results.append(row)
    pd.DataFrame(results).to_csv(output, index=False)

outcomes = ['loneliness_direct_bin', 'loneliness_ucla_total', 'loneliness_djg_total']
compute_missingness(df, ['country_group', 'income_group', 'education_group'], outcomes)

# --- Weighted Summary Statistics ---
summary = df.groupby(['country_group', 'income_group', 'education_group'], observed=True).apply(
    lambda x: pd.Series({
        'n': len(x),
        'n_weighted': x['weight'].sum(),
        'loneliness_direct': np.average(x['loneliness_direct_bin'].dropna(), weights=x.loc[x['loneliness_direct_bin'].notna(), 'weight']) if x['loneliness_direct_bin'].notna().any() else np.nan,
        'loneliness_ucla': np.average(x['loneliness_ucla_total'].dropna(), weights=x.loc[x['loneliness_ucla_total'].notna(), 'weight']) if x['loneliness_ucla_total'].notna().any() else np.nan,
        'loneliness_djg': np.average(x['loneliness_djg_total'].dropna(), weights=x.loc[x['loneliness_djg_total'].notna(), 'weight']) if x['loneliness_djg_total'].notna().any() else np.nan,
    })
).reset_index()
summary.to_csv("income_education_summary_weighted_means.csv", index=False)

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

plot_weighted_bar(df, 'income_group', 'loneliness_direct_bin', 'country_group', "Direct Loneliness by Income", "income_loneliness_direct.png")
plot_weighted_bar(df, 'education_group', 'loneliness_direct_bin', 'country_group', "Direct Loneliness by Education", "education_loneliness_direct.png")
plot_weighted_bar(df, 'income_group', 'loneliness_ucla_total', 'country_group', "UCLA Score by Income", "income_ucla.png")
plot_weighted_bar(df, 'education_group', 'loneliness_ucla_total', 'country_group', "UCLA Score by Education", "education_ucla.png")
plot_weighted_bar(df, 'income_group', 'loneliness_djg_total', 'country_group', "DJG Score by Income", "income_djg.png")
plot_weighted_bar(df, 'education_group', 'loneliness_djg_total', 'country_group', "DJG Score by Education", "education_djg.png")

# --- Forest and Residual Plots ---
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

# --- Weighted Regression (with interaction), Forest/Residuals, Post-hoc ---
def run_income_education_model(data, label, outcome_var):
    model_df = data.dropna(subset=[
        outcome_var, 'income_group', 'education_group', 'age_band', 'municipality_type_group', 'weight'
    ])
    model = smf.wls(
        f"{outcome_var} ~ C(income_group) * C(education_group) + C(age_band) + C(municipality_type_group)",
        data=model_df,
        weights=model_df['weight']
    ).fit()
    with open(f"income_education_{label}_{outcome_var}.txt", "w") as f:
        f.write(model.summary().as_text())
    plot_residuals(model, f"income_education_{label}_{outcome_var}_resid.png", f"Residuals: {label}, {outcome_var}")
    return model, model_df

for group, label in [("Ireland", "ireland"), ("EU", "eu")]:
    data = df[df['country_group'] == group]
    models, model_labels = [], []
    for outcome in ['loneliness_direct_bin', 'loneliness_ucla_total', 'loneliness_djg_total']:
        model, model_df = run_income_education_model(data, label, outcome)
        models.append(model)
        model_labels.append(f"{label}-{outcome}")

        # Post-hoc comparisons (main effects only)
        for factor in ['income_group', 'education_group']:
            valid = model_df.dropna(subset=[outcome, factor])
            if valid[factor].nunique() > 1:
                try:
                    posthoc = pairwise_tukeyhsd(
                        endog=valid[outcome],
                        groups=valid[factor],
                        alpha=0.05
                    )
                    with open(f"posthoc_{factor}_{label}_{outcome}.txt", "w") as f:
                        f.write(str(posthoc.summary()))
                except Exception as e:
                    print(f"Posthoc failed for {factor}-{label}-{outcome}: {e}")
    plot_forest(models, model_labels, f"income_education_{label}_forest_plot.png")

# --- Weighted Chi-Square Tests (Direct binary only) ---
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
    weighted_chi_square(data, 'income_group', 'loneliness_direct_bin', 'weight', label)
    weighted_chi_square(data, 'education_group', 'loneliness_direct_bin', 'weight', label)
