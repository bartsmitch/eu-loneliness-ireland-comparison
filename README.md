# README

## Overview

This repository contains the full analytical workflow and scripts for a quantitative comparison of loneliness in Ireland and the EU27, based on the 2022 EU Loneliness Survey. It covers data preparation, variable coding, weighting, statistical modeling, and visualizations across major sociodemographic and contextual themes.

## Contents

* **EU\_data.xlsx** – Core dataset (not provided here)
* **age\_v\_loneliness\_analysis.py** – Age-based analysis scripts
* **migration\_v\_loneliness\_analysis.py** – Migration status analysis
* **municipality\_v\_loneliness\_analysis.py** – Municipality/urbanicity analysis
* **income\_education\_v\_loneliness\_analysis.py** – Income & education analysis
* **sexuality\_v\_loneliness\_analysis.py** – Sexuality and loneliness analysis
* **technology\_v\_loneliness\_analysis.py** – Social media/technology use analysis
* **Methodology.docx** – Full methodology and workflow
* **eu\_loneliness\_survey\_quest\_eu27.pdf** – Official EU27 survey instrument

## Methodology

* Based on the EU Loneliness Survey (JRC, 2022; N=25,646; 27 countries, including Ireland)
* Weighted cross-sectional design using national/EU weights
* Standardized loneliness outcomes: direct self-report, UCLA scale, De Jong Gierveld scale
* Thematic analyses: age, migration, municipality, income, education, sexuality, technology
* Weighted linear regression (WLS), ANOVA, chi-square, and post-hoc (Tukey) tests
* All outputs: summary tables, regression results, weighted plots

## Usage

1. Place **EU\_data.xlsx** in a `Data/` subfolder.
2. Run each script as needed for analysis of specific themes.
3. Outputs (CSV, PNG, TXT) will be saved to the project directory.
4. See **Methodology.docx** for detailed analytical workflow.

## Requirements

* Python 3.x
* pandas, numpy, statsmodels, matplotlib, seaborn, scipy

## Citation

For data and survey documentation, cite:
European Commission Joint Research Centre (2022). EU Loneliness Survey.
For analytical methods, cite this repository and included methodology file.

## License

For academic/research use only. No redistribution of raw survey data.
