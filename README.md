# Individual Task 2 – Victorian Electricity Demand Analysis

This repository contains the additional analysis completed for Individual Task 2 in the Case Studies in Data Science course.

The work extends my Individual Task 1 project, where I developed models to forecast Victorian electricity demand using historical electricity-demand information, calendar variables and weather data.

For Task 2, I have extended the original analysis to examine the robustness of the modelling approach, performance under different training-set sizes, and differences in model performance across different operating conditions.

---

## Project Overview

The original Task 1 analysis compared two regression models:

- Decision Tree Regressor
- Neural Network using `MLPRegressor`

For Task 2, I extended the analysis to examine:

- chronological / temporal validation
- model performance across different training-set sizes
- operational performance differences using Microsoft Fairlearn
- security, privacy and ethical considerations related to wider deployment

The same processed model-ready dataset and model settings from Task 1 are reused so that the additional analysis remains consistent with the original project.

---

## Dataset

The analysis uses the processed model-ready dataset created during Individual Task 1:

`outputs/stage_03_model_ready_data.csv`

The dataset contains Victorian electricity-demand observations from 2014 together with the features prepared during the Task 1 preprocessing stage.

The prediction target is:

`TOTALDEMAND`

The processed dataset contains lag-based electricity-demand features, calendar features and weather variables.

### Demand-Based Features

The historical electricity-demand predictors include:

- `vk_lag_48`
- `vk_lag_96`
- `vk_lag_336`
- `vk_previous_day_average`
- `vk_previous_7day_average`

These features provide information about previous electricity-demand behaviour.

### Calendar Features

The calendar-related predictors include:

- `vk_is_weekend`
- `vk_hour_sin`
- `vk_hour_cos`
- `vk_day_sin`
- `vk_day_cos`
- `vk_month_sin`
- `vk_month_cos`

The sine and cosine variables are used to represent recurring time patterns such as hour-of-day, day and month.

### Weather Features

The weather predictors used in the models are:

- `vk_min_temp`
- `vk_max_temp`
- `vk_rainfall`
- `vk_humidity_9am`
- `vk_humidity_3pm`
- `vk_temp_9am`
- `vk_temp_3pm`

The electricity-demand information was originally obtained from the Australian Energy Market Operator (AEMO).

The weather information used in the original Task 1 preprocessing was based on Australian weather observations. The processed weather variables are already included in the model-ready dataset used in this repository.

The raw data preparation was completed during Task 1. Task 2 therefore reads the processed model-ready dataset directly instead of repeating the original preprocessing workflow.

---

## Task 2 Analysis

### 1. Temporal Validation

Electricity demand is time-series data, so the observations should not be randomly mixed between training and validation sets.

For this reason, I used an expanding-window chronological validation approach.

The validation periods were:

- July 2014
- August 2014
- September 2014
- October 2014

For each validation period, only observations occurring before that month were used for training.

The November–December period was kept separate as the final unseen test period.

This analysis was used to check whether the models produced consistent results across different time periods.

The generated files are:

- `task2_outputs/task2_01_temporal_cv_results.csv`
- `task2_outputs/task2_02_temporal_cv_summary.csv`
- `task2_outputs/task2_03_temporal_cv_mae.png`

---

### 2. Training-Set Size Analysis

I also examined how model performance changed when different amounts of historical training data were available.

October was kept as the fixed validation period, while the amount of earlier chronological training data was varied.

The following training proportions were tested:

- 20%
- 40%
- 60%
- 80%
- 100%

Because the subsets were selected chronologically, increasing the training-set size also changed the historical and seasonal coverage available to the models.

For this reason, the experiment is interpreted as a training-size sensitivity analysis rather than assuming that changes in performance were caused only by the number of observations.

The generated files are:

- `task2_outputs/task2_04_learning_curve_results.csv`
- `task2_outputs/task2_05_learning_curve.png`

---

### 3. Fairlearn Analysis

Microsoft Fairlearn `MetricFrame` was used to compare model prediction errors across different operating conditions.

The dataset does not contain demographic protected attributes such as race, gender or age. Therefore, this part of the analysis is not intended to measure demographic fairness.

Instead, Fairlearn is used to examine operational differences in model performance.

The groups examined were:

- weekday vs weekend observations
- different maximum-temperature ranges

The temperature groups used were:

- less than or equal to 15°C
- 15–25°C
- 25–30°C
- above 30°C

Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) were calculated for the different groups.

The analysis also calculates the difference between the highest and lowest group MAE to identify whether model performance varies considerably across operating conditions.

The generated files are:

- `task2_outputs/task2_06_weekend_fairness.csv`
- `task2_outputs/task2_07_temperature_fairness.csv`
- `task2_outputs/task2_08_fairness_error_gaps.csv`

---

## Models

### Decision Tree

The Decision Tree model uses:

```text
max_depth = 15
min_samples_leaf = 100
random_state = 42
