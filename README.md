# Piercing the Sky Responsibly: Predicting NYC Skyscrapers’ Carbon Compliance

**Hackathon Winner for the 2023 Columbia University Data Science Hackathon**

**TL;DR.** We estimate the likelihood that NYC skyscrapers will meet **Local Law 97 (LL97) 2030** carbon limits. Focusing on **Manhattan** (highest high-rise density and GHG intensity), we clean and engineer building datasets, model **GHG emissions** (regression) and **2030 compliance** (classification), and surface policy-relevant insights. Best regression: **Ridge (scaled & tuned)** ≈ **68% accuracy**. Compliance classification: **Logistic Regression** ≈ **62% accuracy**. **Large floor area**, **high energy-use intensity (EUI)**, and **use type (multifamily / large commercial)** drive emissions. Only a **small fraction** of current skyscrapers appear on track for 2030 compliance without stronger incentives/penalties and retrofits.

---

## Project Overview

**Context.** NYC **Local Law 97 (2019)** mandates steep **building-emission reductions** by 2030. Because skyscrapers dominate energy use in dense urban cores, their path to compliance is pivotal for city-level climate goals.

**Goals.**
- Quantify **drivers of GHG emissions** for high-rise buildings.
- Predict **absolute emissions** and **binary compliance** proxies for 2030.
- Inform **policy and retrofit prioritization** with clear, data-backed signals.

**Manhattan focus.** Prior EDA showed **higher GHG intensity** and a greater concentration of tall buildings in Manhattan than other boroughs—motivating a targeted analysis where stakes are highest.

---

## Data, Cleaning & Preparation

- **Scope & sources.** Building-level features (e.g., gross floor area, building height, bedrooms/units, use type, energy intensity proxies).  
- **Imputation.** **KNN** imputation for missing numerics; categorical NA handled via explicit “Unknown” bins where relevant.  
- **Feature selection.** Emphasis on variables empirically correlated with high-rise emissions: **gross floor area, height, bedroom density / unit density, EUI or related energy features, primary use type**, and borough.  
- **Transformations.**  
  - Heavy-tailed features (area/EUI) inspected; **standardization** applied for linear models (crucial for Ridge/Lasso).  
  - Optional log transforms were evaluated; retained only where they improved cross-validated error.  
- **Train/validation/test.** Stratified splits by use-type/size bands to avoid leakage from extremely similar buildings and to preserve class balance for compliance labels.

---

## Exploratory Data Analysis (highlights)

- **Intensity gradient.** Manhattan exhibits the **highest per-building emissions intensity** among boroughs, consistent with its **height + area** profile.  
- **Correlations.** **Floor area** is the dominant linear correlate with total GHG, followed by **EUI** and (to a lesser degree) **height**. Multifamily and large commercial clusters sit at the upper tail.  
- **Outliers & leverage.** A few ultra-large assets act as **leverage points**—a key reason regularization (Ridge) beats vanilla OLS.

---

## Modeling & Evaluation

### 1) Emissions Regression
**Models.** KNN Regressor, OLS, **Ridge**, Lasso (all with scaling where appropriate).  
**Tuning.** Hyperparameters selected via nested CV (grid over α for Ridge/Lasso; neighbors/distance metric for KNN).  
**Result.** **Ridge (scaled+tuned)** achieves **≈ 68% accuracy** (notebook details include metric choice & CV folds).  
**Takeaways.**  
- **Collinearity** (area, height, usage patterns) makes **Ridge** robust vs. OLS.  
- **Feature weights** (standardized) highlight **gross floor area** and **EUI** as the strongest positive contributors; **use type** effects show **multifamily** and **large commercial** consistently higher.  
- **KNN** performs competitively on mid-range assets but is sensitive to local density in feature space; Lasso sparsifies but underfits at extremes.

### 2) Compliance Classification (2030 Proxy)
**Labeling.** “Compliant” approximated as **below national medians / policy-aligned thresholds** for GHG intensity (proxy for 2030 caps).  
**Model.** **Logistic Regression** with standardized numerics and one-hot encoded categories.  
**Performance.** **≈ 62% accuracy** with reasonable precision/recall balance (see notebook for PR curves, calibration, and threshold sweeps).  
**Finding.** Under current performance distributions, **only a small percentage** of skyscrapers appear **on track** for 2030 compliance—implying meaningful shortfalls without intervention.

**Evaluation notes.**
- We report accuracy for headline comparability and include **MAE/R²** for regression and **AUROC/PR** for classification in the notebook.
- **Calibration** checked; logistic model is slightly under-confident—temperature scaling marginally improves Brier score but not headline accuracy.

---

## Policy-Relevant Insights

- **Concentration risk.** A relatively small set of very large assets contributes a disproportionate share of absolute GHG. Targeting these with **deep retrofits** yields outsized impact.  
- **Multifamily & large commercial** are the **heaviest segments**; tailored programs (e.g., electrification incentives, EUI reduction via HVAC upgrades, envelope improvements) are likely necessary.  
- **Standards + incentives.** Penalties alone risk non-compliance and pass-through costs; **incentive structures** (rebates, low-interest retrofit finance) can accelerate upgrades.  
- **Data gaps.** Incorporating **fuel mix / on-site generation**, **retrofit history**, **system age**, and **occupancy patterns** should sharpen predictions and retrofit ROI estimates.

---

## Limitations & Future Work

- **Proxy labels.** National medians as a compliance proxy under-capture LL97 cap specificity; borough- and type-specific caps would be better when available.  
- **Static snapshot.** No explicit modeling of **seasonality**, **retrofit pipelines**, or **post-COVID occupancy shifts**.  
- **Next steps.**  
  - **Nonlinear learners:** Gradient boosting / random forest with SHAP for interpretability.  
  - **Hierarchical effects:** Mixed models by borough / use-type.  
  - **Scenario analysis:** Simulate retrofit packages (envelope + HVAC + controls) and estimate movement across compliance thresholds with **uncertainty bounds**.  
  - **Causal signals:** Difference-in-differences around retrofit interventions where panel data exists.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate    # or: conda create -n ll97 python=3.11
python -m pip install -r requirements.txt
# Data handling:
#  - Private (default): put raw/processed files in data/ (gitignored)
#  - Public (optional): small, shareable CSVs in data_public/ with a README noting source & license
jupyter notebook notebooks/NYC_LL97_PiercingTheSky.ipynb
# Optional: export an executed HTML for quick viewing
jupyter nbconvert --to html --execute notebooks/NYC_LL97_PiercingTheSky.ipynb \
  --output ./assets/notebook.html --ExecutePreprocessor.timeout=600
```

---
## Contributors:
Layanne El Assaad, Nicholas Ding Yang Choong, Tawab Safi
