# Individual Task 2 – Victorian Electricity Demand Analysis

This repository contains the additional analysis completed for Individual Task 2 in the Case Studies in Data Science course.

The work extends my Individual Task 1 project, which focused on forecasting Victorian electricity demand using historical demand and weather data.

## Project Overview

The original Task 1 analysis compared two regression models:

- Decision Tree Regressor
- Neural Network using `MLPRegressor`

For Task 2, I extended the analysis to examine:

- chronological / temporal validation
- model performance across different training-set sizes
- operational subgroup differences using Microsoft Fairlearn
- security, privacy and ethical considerations related to wider deployment

## Task 2 Analysis

### 1. Temporal Validation

Because electricity demand is time-series data, the analysis preserves chronological order instead of using random K-fold cross-validation.

An expanding-window temporal validation approach was used across:

- July
- August
- September
- October

The November–December period remained separate as the final unseen test period.

The purpose of this analysis was to evaluate whether the Decision Tree and Neural Network produced consistent performance across different time periods.

### 2. Training-Set Size Analysis

The effect of training-set size was investigated by keeping October as the validation period and varying the amount of earlier chronological training data.

The following training proportions were tested:

- 20%
- 40%
- 60%
- 80%
- 100%

This analysis was used to examine how model generalisation changed as more historical observations became available.

Because the subsets were chronological, changes in training size also changed the temporal and seasonal coverage of the training data.

### 3. Fairlearn Analysis

Microsoft Fairlearn `MetricFrame` was used to compare prediction errors across different operating conditions.

The dataset does not contain protected demographic attributes such as race, gender or age. Therefore, this analysis focuses on operational performance disparities rather than demographic fairness.

The groups examined were:

- weekday vs weekend
- temperature ranges

This analysis helped identify whether model performance was consistent across different parts of the final test data.

### 4. Security, Privacy and Ethical Considerations

The Task 2 reflection also considers possible risks if the forecasting approach were deployed more widely.

These include:

- integrity of electricity-demand and weather inputs
- risks from corrupted or manipulated data
- privacy concerns if household-level smart-meter data were used
- limitations caused by underrepresented operating conditions
- transfer of a model trained on historical data to different years or locations
- the importance of human oversight in operational decision-making

## Dataset Used

Task 2 uses the same model-ready dataset prepared in Individual Task 1:

`outputs/stage_03_model_ready_data.csv`

This file contains the processed features required by the forecasting models, including historical demand information, lag-based predictors, calendar variables and weather features.

The Task 2 script reads this file directly so that the additional temporal-validation, training-size and Fairlearn analyses remain consistent with the original Task 1 modelling pipeline.

## Repository Structure

```text
task2-analysis/
│
├── README.md
├── vk_task2_analysis.py
├── requirements.txt
│
├── outputs/
│   └── stage_03_model_ready_data.csv
│
├── results/
│   ├── task2_01_temporal_cv_results.csv
│   ├── task2_02_temporal_cv_summary.csv
│   ├── task2_04_learning_curve_results.csv
│   ├── task2_06_weekend_fairness.csv
│   ├── task2_07_temperature_fairness.csv
│   └── task2_08_fairness_error_gaps.csv
│
└── figures/
    ├── task2_03_temporal_cv_mae.png
    └── task2_05_learning_curve.png
```

## Output Files

### Temporal Validation

- `task2_01_temporal_cv_results.csv`
- `task2_02_temporal_cv_summary.csv`
- `task2_03_temporal_cv_mae.png`

### Training-Set Size Analysis

- `task2_04_learning_curve_results.csv`
- `task2_05_learning_curve.png`

### Fairlearn Analysis

- `task2_06_weekend_fairness.csv`
- `task2_07_temperature_fairness.csv`
- `task2_08_fairness_error_gaps.csv`

## Requirements

The main Python packages used are:

```text
pandas
numpy
scikit-learn
matplotlib
fairlearn
```

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Running the Analysis

Run the Task 2 analysis script using:

```bash
python vk_task2_analysis.py
```

The script reads:

```text
outputs/stage_03_model_ready_data.csv
```

and generates the temporal-validation, training-size and Fairlearn results used in the Task 2 report.

## Data Sources

The project uses Victorian electricity-demand data together with Victorian weather information.

The original electricity-demand data were obtained from AEMO, while the weather information was obtained from the Bureau of Meteorology.

The processed model-ready file included in this repository was prepared during Individual Task 1 and reused in Task 2 to maintain consistency with the original forecasting analysis.

## Important Limitations

The analysis has several limitations that should be considered when interpreting the results:

- model performance changes across different time periods
- some operating conditions are underrepresented
- the final test period contains no observations at or below 15°C
- historical 2014 data may not represent future demand behaviour
- weather information was averaged across multiple Victorian stations
- observed weather was used rather than future weather forecasts

For these reasons, the results should not be interpreted as evidence that the models will perform equally well under all future operating conditions.

## Academic Use

This repository was created for academic coursework.

The code and outputs should be interpreted in the context of the associated assessment report and its stated assumptions and limitations.
