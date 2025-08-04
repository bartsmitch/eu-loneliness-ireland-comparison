import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --- Plotting functions ---
def plot_weighted_bar(df, y_col, title, filename):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df[df[y_col].notna()],
        x='age_band',
        y=y_col,
        hue='country_group',
        estimator=lambda y: np.average(y, weights=df.loc[y.index, 'weight']),
        errorbar=None
    )
    plt.title(title)
    plt.ylabel("Weighted Mean")
    plt.xlabel("Age Band")
    plt.legend(title='Country')
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
    df_plot = pd.DataFrame(rows).sort_values(by='Variable')
    plt.figure(figsize=(12, 2 + 0.5 * len(df_plot)))
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

# --- Load Data ---
df = pd.read_excel("Data/EU_data.xlsx")

# --- Data Preparation ---
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df = df[df['age'].between(16, 120)]
age_bins = [16, 24, 34, 44, 54, 64, np.inf]
age_labels = ['16–24','25–34','35–44','45–54','55–64','65+']
df['age_band'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True)
df['country_group'] = df['country'].apply(lambda x: 'Ireland' if x == 'Ireland' else 'EU')
df['weight'] = np.where(df['country_group'] == 'Ireland', df['w_country'], df['w_eu27'])

# --- Loneliness Variables ---
loneliness_direct_map = {
    "All of the time": 1,
    "Most of the time": 1,
    "Some of the time": 1,
    "A little of the time": 0,
    "None of the time": 0
}
df['loneliness_direct_binary'] = df['loneliness_direct'].map(loneliness_direct_map)
ucla_map = {
    "Hardly ever or never": 1,
    "Some of the time": 2,
    "Often": 3
}
for item in ['loneliness_ucla_a', 'loneliness_ucla_b', 'loneliness_ucla_c']:
    df[item + '_score'] = df[item].map(ucla_map)
df['loneliness_ucla_total'] = df[['loneliness_ucla_a_score', 'loneliness_ucla_b_score', 'loneliness_ucla_c_score']].sum(axis=1)

neg_items = ['loneliness_djg_a', 'loneliness_djg_b', 'loneliness_djg_c']
pos_items = ['loneliness_djg_d', 'loneliness_djg_e', 'loneliness_djg_f']
for item in neg_items:
    df[item + '_recoded'] = df[item].map({'Yes': 1, 'More or less': 1, 'No': 0})
for item in pos_items:
    df[item + '_recoded'] = df[item].map({'Yes': 0, 'More or less': 1, 'No': 1})
djg_recode_cols = [i + '_recoded' for i in neg_items + pos_items]
df['loneliness_djg_total'] = df[djg_recode_cols].sum(axis=1)

# --- Weighted Prevalence/Means ---
for col in ['loneliness_direct_binary', 'loneliness_ucla_total', 'loneliness_djg_total']:
    plot_weighted_bar(df, col, f"Weighted {col.replace('_',' ').title()} by Age Band and Country", f"{col}_age_country.png")

summary = df.groupby(['country_group', 'age_band'], observed=True).apply(
    lambda x: pd.Series({
        'n': len(x),
        'loneliness_direct_mean': np.average(x['loneliness_direct_binary'].dropna(), weights=x.loc[x['loneliness_direct_binary'].notna(), 'weight']),
        'loneliness_ucla_mean': np.average(x['loneliness_ucla_total'].dropna(), weights=x.loc[x['loneliness_ucla_total'].notna(), 'weight']),
        'loneliness_djg_mean': np.average(x['loneliness_djg_total'].dropna(), weights=x.loc[x['loneliness_djg_total'].notna(), 'weight'])
    })
).reset_index()
summary.to_csv("age_summary_weighted_means.csv", index=False)

# --- Cross-Tabs ---
for col in ['loneliness_direct_binary', 'loneliness_ucla_total', 'loneliness_djg_total']:
    tab = df[df[col].notna()].pivot_table(
        index='age_band',
        columns='country_group',
        values=col,
        aggfunc=lambda x: np.average(x, weights=df.loc[x.index, 'weight'])
    )
    tab.to_csv(f"age_cross_tab_{col}.csv")

# --- Weighted ANOVA and Post-hoc (Ireland, EU, Interaction) ---
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def run_anova_and_posthoc(df, label, outcome):
    # Main effect
    anova_df = df.dropna(subset=[outcome, 'age_band', 'weight'])
    model = smf.wls(f'{outcome} ~ C(age_band)', data=anova_df, weights=anova_df['weight']).fit()
    with open(f"anova_{label}_{outcome}.txt", "w") as f:
        f.write(model.summary().as_text())
    plot_forest([model], [label], f"anova_{label}_{outcome}_coefplot.png")
    plot_residuals(model, f"anova_{label}_{outcome}_resid.png", f"Residuals: {label}, {outcome}")
    # Post-hoc
    valid = anova_df.dropna(subset=[outcome, 'age_band'])
    if valid['age_band'].nunique() > 1:
        try:
            posthoc = pairwise_tukeyhsd(
                endog=valid[outcome],
                groups=valid['age_band'],
                alpha=0.05
            )
            with open(f"posthoc_{label}_{outcome}.txt", "w") as f:
                f.write(str(posthoc.summary()))
        except Exception as e:
            print(f"Posthoc failed for {label}-{outcome}: {e}")
    return model

models, labels = [], []
for group, label in [("Ireland", "Ireland"), ("EU", "EU")]:
    dfg = df[df['country_group'] == group]
    model = run_anova_and_posthoc(dfg, label, 'loneliness_direct_binary')
    models.append(model)
    labels.append(label)

# --- Interaction model (age_band * country_group) ---
model_df = df.dropna(subset=['loneliness_direct_binary', 'age_band', 'country_group', 'weight'])
interaction_model = smf.wls('loneliness_direct_binary ~ C(age_band) * C(country_group)', data=model_df, weights=model_df['weight']).fit()
with open("anova_interaction.txt", "w") as f:
    f.write(interaction_model.summary().as_text())
plot_forest([interaction_model], ["Interaction"], "anova_interaction_coefplot.png")
plot_residuals(interaction_model, "anova_interaction_resid.png", "Residuals: Interaction Model")

# --- Summary table for ANOVA ---
anova_results = pd.DataFrame({
    'Model': ['Ireland ANOVA', 'EU ANOVA', 'Interaction Model'],
    'F-statistic': [models[0].fvalue, models[1].fvalue, interaction_model.fvalue],
    'p-value': [models[0].f_pvalue, models[1].f_pvalue, interaction_model.f_pvalue],
    'R-squared': [models[0].rsquared, models[1].rsquared, interaction_model.rsquared],
    'Adj. R-squared': [models[0].rsquared_adj, models[1].rsquared_adj, interaction_model.rsquared_adj]
})
anova_results.to_csv("anova_summary_table.csv", index=False)
