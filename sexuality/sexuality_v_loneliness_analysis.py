import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data ---
df = pd.read_excel("Data/EU_data.xlsx")

# --- Data Preparation ---

# Sexual orientation group
orientation_map = {
    "Heterosexual/ straight": "Heterosexual",
    "Lesbian or gay": "Homosexual",
    "Bisexual": "Bisexual",
    "Other sexual orientation": "Other",
    "Prefer not to say": "Prefer not to say"
}
df['sex_orientation_group'] = df['sex_orientation'].map(orientation_map)
df = df[df['sex_orientation_group'] != "Prefer not to say"]

# Age banding
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df = df[df['age'].between(16, 120)]
age_bins = [16, 24, 34, 44, 54, 64, np.inf]
age_labels = ['16–24', '25–34', '35–44', '45–54', '55–64', '65+']
df['age_band'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True)

# Migration
df['migrant_flag'] = (
    (df['country_birth_yourself'] != df['country']) |
    (df['country_birth_mother'] != df['country']) |
    (df['country_birth_father'] != df['country'])
).astype(int)

# Income group
income_map = {
    'Decile 1': 'Low', 'Decile 2': 'Low', 'Decile 3': 'Low',
    'Decile 4': 'Medium', 'Decile 5': 'Medium', 'Decile 6': 'Medium', 'Decile 7': 'Medium',
    'Decile 8': 'High', 'Decile 9': 'High', 'Decile 10': 'High'
}
df['income_group'] = df['income_decile'].map(income_map)

# Weighting and country group
df['country_group'] = df['country'].apply(lambda x: 'Ireland' if x == 'Ireland' else 'EU')
df['weight'] = df.apply(lambda row: row['w_country'] if row['country_group'] == 'Ireland' else row['w_eu27'], axis=1)

# --- Loneliness Variables ---
direct_map = {
    "All of the time": 1,
    "Most of the time": 1,
    "Some of the time": 1,
    "A little of the time": 0,
    "None of the time": 0
}
df['loneliness_direct_bin'] = df['loneliness_direct'].map(direct_map)
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

# --- Summary Statistics ---
summary = df.groupby(['country_group', 'sex_orientation_group'], observed=True).apply(
    lambda x: pd.Series({
        'n': len(x),
        'n_weighted': x['weight'].sum(),
        'loneliness_direct': np.average(x['loneliness_direct_bin'].dropna(), weights=x.loc[x['loneliness_direct_bin'].notna(), 'weight']) if x['loneliness_direct_bin'].notna().any() else np.nan,
        'loneliness_ucla': np.average(x['loneliness_ucla_total'].dropna(), weights=x.loc[x['loneliness_ucla_total'].notna(), 'weight']) if x['loneliness_ucla_total'].notna().any() else np.nan,
        'loneliness_djg': np.average(x['loneliness_djg_total'].dropna(), weights=x.loc[x['loneliness_djg_total'].notna(), 'weight']) if x['loneliness_djg_total'].notna().any() else np.nan,
    })
).reset_index()
summary.to_csv("sexuality_summary_weighted_means.csv", index=False)

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

plot_weighted_bar(df, 'sex_orientation_group', 'loneliness_direct_bin', 'country_group', "Direct Loneliness by Sexual Orientation", "sexuality_loneliness_direct.png")
plot_weighted_bar(df, 'sex_orientation_group', 'loneliness_ucla_total', 'country_group', "UCLA Score by Sexual Orientation", "sexuality_ucla.png")
plot_weighted_bar(df, 'sex_orientation_group', 'loneliness_djg_total', 'country_group', "DJG Score by Sexual Orientation", "sexuality_djg.png")

# --- Forest/Residual Plots ---
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

# --- Weighted Regression with Plots ---
def run_sexuality_model(data, label, outcome_var):
    model_df = data.dropna(subset=[
        outcome_var, 'sex_orientation_group', 'age_band', 'migrant_flag', 'income_group', 'weight'
    ])
    model = smf.wls(
        f"{outcome_var} ~ C(sex_orientation_group) + C(age_band) + migrant_flag + C(income_group)",
        data=model_df,
        weights=model_df['weight']
    ).fit()
    with open(f"sexuality_{label}_{outcome_var}.txt", "w") as f:
        f.write(model.summary().as_text())
    plot_residuals(model, f"sexuality_{label}_{outcome_var}_resid.png", f"Residuals: {label}, {outcome_var}")
    return model

for group, label in [("Ireland", "ireland"), ("EU", "eu")]:
    data = df[df['country_group'] == group]
    models, labels_ = [], []
    for outcome in ['loneliness_direct_bin', 'loneliness_ucla_total', 'loneliness_djg_total']:
        models.append(run_sexuality_model(data, label, outcome))
        labels_.append(f"{label}-{outcome}")
    plot_forest(models, labels_, f"sexuality_{label}_forest_plot.png")

# --- Interaction analysis with plots ---
def run_interaction_models(data, label, outcome_var):
    model_df = data.dropna(subset=[
        outcome_var, 'sex_orientation_group', 'age_band', 'migrant_flag', 'income_group', 'weight'
    ])
    interaction_vars = ['age_band', 'migrant_flag', 'income_group']
    for ivar in interaction_vars:
        formula = f"{outcome_var} ~ C(sex_orientation_group) * C({ivar}) + C(age_band) + migrant_flag + C(income_group)"
        model = smf.wls(formula, data=model_df, weights=model_df['weight']).fit()
        fname = f"sexuality_{label}_{outcome_var}_interaction_{ivar}.txt"
        with open(fname, "w") as f:
            f.write(model.summary().as_text())
        plot_residuals(model, f"sexuality_{label}_{outcome_var}_interaction_{ivar}_resid.png", f"Residuals: {label}, {outcome_var} * {ivar}")
        plot_forest([model], [f"{label}-{outcome_var}*{ivar}"], f"sexuality_{label}_{outcome_var}_interaction_{ivar}_coefplot.png")

        # --- Interaction Plot ---
        if ivar == 'age_band' or ivar == 'income_group':
            for group_val in sorted(model_df[ivar].dropna().unique()):
                subset = model_df[model_df[ivar] == group_val]
                plt.figure(figsize=(10, 6))
                sns.barplot(
                    data=subset,
                    x='sex_orientation_group',
                    y=outcome_var,
                    estimator=lambda y: np.average(y, weights=subset.loc[y.index, 'weight']),
                    errorbar=None
                )
                plt.title(f"{outcome_var.replace('_',' ').title()} by Sexual Orientation and {ivar.replace('_',' ').title()} ({group_val})")
                plt.ylabel("Weighted Mean")
                plt.xlabel("Sexual Orientation")
                plt.tight_layout()
                plt.savefig(f"sexuality_{label}_{outcome_var}_interaction_{ivar}_{group_val}.png")
                plt.close()
        elif ivar == 'migrant_flag':
            for group_val in [0,1]:
                subset = model_df[model_df[ivar] == group_val]
                plt.figure(figsize=(10, 6))
                sns.barplot(
                    data=subset,
                    x='sex_orientation_group',
                    y=outcome_var,
                    estimator=lambda y: np.average(y, weights=subset.loc[y.index, 'weight']),
                    errorbar=None
                )
                lab = "Non-migrant" if group_val == 0 else "Migrant"
                plt.title(f"{outcome_var.replace('_',' ').title()} by Sexual Orientation and {lab}")
                plt.ylabel("Weighted Mean")
                plt.xlabel("Sexual Orientation")
                plt.tight_layout()
                plt.savefig(f"sexuality_{label}_{outcome_var}_interaction_{ivar}_{lab}.png")
                plt.close()

for group, label in [("Ireland", "ireland"), ("EU", "eu")]:
    data = df[df['country_group'] == group]
    for outcome in ['loneliness_direct_bin', 'loneliness_ucla_total', 'loneliness_djg_total']:
        run_interaction_models(data, label, outcome)

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
    weighted_chi_square(data, 'sex_orientation_group', 'loneliness_direct_bin', 'weight', label)
