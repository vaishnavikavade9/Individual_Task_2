"""
student id - s4149698
name - vaishnavi vishal kavade
case studies with data science - individual task 2

this script is extending my task 1 electricity demand case study

here i am gonna check:
1. temporal cross validation
2. model performance with different training sizes
3. model performance differences across meaningful groups
"""

# importing the needed libraries

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# fairlearn is gonna help me compare model errors across different groups
from fairlearn.metrics import MetricFrame


# project paths

# this is gonna give me the folder where this task 2 script is located
vk_project_folder = Path(__file__).resolve().parent

# i am gonna use the model ready dataset already created in task 1
vk_model_ready_file = (
    vk_project_folder
    / "outputs"
    / "stage_03_model_ready_data.csv"
)

# now i am gonn kip task 2 outputs separate from task 1 outputs
vk_task2_output_folder = (
    vk_project_folder
    / "task2_outputs"
)

# lets create the task 2 output folder if it is not already there
vk_task2_output_folder.mkdir(
    exist_ok=True
)


# helper functions

# this small function is gonna hlepp me print clear headings for each stage
def vk_print_heading(vk_title):

    print(
        "\n"
        + "=" * 80
    )

    print(
        vk_title
    )

    print(
        "=" * 80
    )


# this function is gonna calculate the seme measures that i used in task 1
def vk_calculate_metrics(
    vk_model_name,
    vk_actual_values,
    vk_predicted_values
):

    # mae is gonna show me the average prediction error in mw
    vk_mae = mean_absolute_error(
        vk_actual_values,
        vk_predicted_values
    )

    # rmse is gonna show the eror too but bigger mistakes would get more weight here
    vk_rmse = np.sqrt(
        mean_squared_error(
            vk_actual_values,
            vk_predicted_values
        )
    )

    # r2 is gonna show how much variation in demand is explained by the model
    vk_r2 = r2_score(
        vk_actual_values,
        vk_predicted_values
    )

    # mape is gonna show the average prediction error as percentage
    vk_mape = (
        mean_absolute_percentage_error(
            vk_actual_values,
            vk_predicted_values
        )
        * 100
    )

    # now i am gonn return all of these together
    return {
        "Model": vk_model_name,
        "MAE_MW": vk_mae,
        "RMSE_MW": vk_rmse,
        "R2": vk_r2,
        "MAPE_percent": vk_mape
    }


# fairlearn needs a normal metric function so i am gonna make one for rmse
def vk_rmse_metric(
    vk_actual_values,
    vk_predicted_values
):

    return np.sqrt(
        mean_squared_error(
            vk_actual_values,
            vk_predicted_values
        )
    )


# this helper is gonna train the neural network while keeping scaling based only on training data
def vk_train_neural_model(
    vk_x_train_data,
    vk_y_train_data,
    vk_x_evaluation_data
):

    # now i am gonna create fresh scalers for this training run
    vk_x_scaler = StandardScaler()
    vk_y_scaler = StandardScaler()

    # lets fit both scalers only on the training information
    vk_x_scaler.fit(
        vk_x_train_data
    )

    vk_y_scaler.fit(
        vk_y_train_data
        .to_numpy()
        .reshape(-1, 1)
    )

    # now i am gonna scale the training predictors
    vk_x_train_scaled = (
        vk_x_scaler
        .transform(
            vk_x_train_data
        )
    )

    # now scaling the target values too
    vk_y_train_scaled = (
        vk_y_scaler
        .transform(
            vk_y_train_data
            .to_numpy()
            .reshape(-1, 1)
        )
        .ravel()
    )

    # now i am gonna use the same training scaler on the evaluation data
    # i am not fitting a new scaler there because that could bring extra information into training
    vk_x_evaluation_scaled = (
        vk_x_scaler
        .transform(
            vk_x_evaluation_data
        )
    )

    # i am gonna use the same neural network settings selected in task 1
    vk_neural_model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.1,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    )

    # now lets train the model
    vk_neural_model.fit(
        vk_x_train_scaled,
        vk_y_train_scaled
    )

    # these predictions would still be on the scaled target
    vk_prediction_scaled = (
        vk_neural_model
        .predict(
            vk_x_evaluation_scaled
        )
    )

    # lets convert the prediction back into actual electricity demand in mw
    vk_prediction = (
        vk_y_scaler
        .inverse_transform(
            vk_prediction_scaled
            .reshape(-1, 1)
        )
        .ravel()
    )

    return (
        vk_neural_model,
        vk_x_scaler,
        vk_y_scaler,
        vk_prediction
    )


# stage 1 - loading the task 1 model ready data

vk_print_heading(
    "STAGE 1 - LOAD THE TASK 1 MODEL READY DATA"
)

# now i am gonna read the exact model ready dataset that came from my task 1
vk_model_data = pd.read_csv(
    vk_model_ready_file
)

# lets convert the date columns back into datetime format
vk_model_data[
    "SETTLEMENTDATE"
] = pd.to_datetime(
    vk_model_data[
        "SETTLEMENTDATE"
    ]
)

vk_model_data[
    "vk_date"
] = pd.to_datetime(
    vk_model_data[
        "vk_date"
    ]
)

# i want everything to stay in the correct time order before i start the experiments
vk_model_data = (
    vk_model_data
    .sort_values(
        "SETTLEMENTDATE"
    )
    .reset_index(
        drop=True
    )
)

print(
    "Model ready observations:",
    len(vk_model_data)
)

print(
    "First date:",
    vk_model_data[
        "vk_date"
    ].min()
)

print(
    "Last date:",
    vk_model_data[
        "vk_date"
    ].max()
)


# stage 2 - keeping the same predictors from task 1

vk_print_heading(
    "STAGE 2 - DEFINE THE SAME TASK 1 PREDICTORS"
)

# these are exactly the predictors i ussed in my original task 1 analysis
vk_predictor_names = [
    "vk_lag_48",
    "vk_lag_96",
    "vk_lag_336",
    "vk_previous_day_average",
    "vk_previous_7day_average",
    "vk_is_weekend",
    "vk_hour_sin",
    "vk_hour_cos",
    "vk_day_sin",
    "vk_day_cos",
    "vk_month_sin",
    "vk_month_cos",
    "vk_min_temp",
    "vk_max_temp",
    "vk_rainfall",
    "vk_humidity_9am",
    "vk_humidity_3pm",
    "vk_temp_9am",
    "vk_temp_3pm"
]

# total demand would again be my prediction target
vk_target_name = "TOTALDEMAND"

print(
    "Number of predictors:",
    len(vk_predictor_names)
)


# stage 3 - setting up temporal cross validation

vk_print_heading(
    "STAGE 3 - TEMPORAL CROSS VALIDATION"
)

# now i am gonna check the model more than one time instead of depending on only one validation period
# as my electricity data is following time order i don't want to randomly mix the observations
# here i would be training the model on previous months and checking it on the next month

vk_cv_periods = [
    (
        "July",
        pd.Timestamp("2014-07-01"),
        pd.Timestamp("2014-08-01")
    ),
    (
        "August",
        pd.Timestamp("2014-08-01"),
        pd.Timestamp("2014-09-01")
    ),
    (
        "September",
        pd.Timestamp("2014-09-01"),
        pd.Timestamp("2014-10-01")
    ),
    (
        "October",
        pd.Timestamp("2014-10-01"),
        pd.Timestamp("2014-11-01")
    )
]

# lets create one empty list where i am gonna keep all the fold results
vk_cv_results = []


# stage 4 - running temporal cross validation

vk_print_heading(
    "STAGE 4 - RUN TEMPORAL CROSS VALIDATION"
)

# now i am gonna go through every validation month one by one
for (
    vk_validation_name,
    vk_validation_start,
    vk_validation_end
) in vk_cv_periods:

    print(
        "\nvalidation period:",
        vk_validation_name
    )

    # now all the observations before this month are gonna be used for training
    # this way the model is only learning from information that would already be available at that time
    vk_cv_train_mask = (
        vk_model_data[
            "vk_date"
        ]
        < vk_validation_start
    )

    # now i am gonna keep only the selected month for this validation fold
    vk_cv_validation_mask = (
        (
            vk_model_data[
                "vk_date"
            ]
            >= vk_validation_start
        )
        &
        (
            vk_model_data[
                "vk_date"
            ]
            < vk_validation_end
        )
    )

    vk_x_cv_train = (
        vk_model_data
        .loc[
            vk_cv_train_mask,
            vk_predictor_names
        ]
    )

    vk_y_cv_train = (
        vk_model_data
        .loc[
            vk_cv_train_mask,
            vk_target_name
        ]
    )

    vk_x_cv_validation = (
        vk_model_data
        .loc[
            vk_cv_validation_mask,
            vk_predictor_names
        ]
    )

    vk_y_cv_validation = (
        vk_model_data
        .loc[
            vk_cv_validation_mask,
            vk_target_name
        ]
    )

    print(
        "training rows:",
        len(vk_x_cv_train)
    )

    print(
        "validation rows:",
        len(vk_x_cv_validation)
    )

    # now i would be training the decision tree using the same settings from task 1
    vk_cv_tree_model = DecisionTreeRegressor(
        max_depth=15,
        min_samples_leaf=100,
        random_state=42
    )

    vk_cv_tree_model.fit(
        vk_x_cv_train,
        vk_y_cv_train
    )

    # lets get the decision tree prediction for this validation month
    vk_cv_tree_prediction = (
        vk_cv_tree_model
        .predict(
            vk_x_cv_validation
        )
    )

    # now lets calculate the errors for this fold
    vk_tree_fold_metrics = (
        vk_calculate_metrics(
            "Decision Tree",
            vk_y_cv_validation,
            vk_cv_tree_prediction
        )
    )

    vk_tree_fold_metrics[
        "Validation_period"
    ] = vk_validation_name

    vk_tree_fold_metrics[
        "Training_rows"
    ] = len(
        vk_x_cv_train
    )

    vk_tree_fold_metrics[
        "Validation_rows"
    ] = len(
        vk_x_cv_validation
    )

    vk_cv_results.append(
        vk_tree_fold_metrics
    )

    # now i am gonn do the same validation month using the neural network
    (
        vk_cv_neural_model,
        vk_cv_x_scaler,
        vk_cv_y_scaler,
        vk_cv_neural_prediction
    ) = vk_train_neural_model(
        vk_x_cv_train,
        vk_y_cv_train,
        vk_x_cv_validation
    )

    vk_neural_fold_metrics = (
        vk_calculate_metrics(
            "Neural Network",
            vk_y_cv_validation,
            vk_cv_neural_prediction
        )
    )

    vk_neural_fold_metrics[
        "Validation_period"
    ] = vk_validation_name

    vk_neural_fold_metrics[
        "Training_rows"
    ] = len(
        vk_x_cv_train
    )

    vk_neural_fold_metrics[
        "Validation_rows"
    ] = len(
        vk_x_cv_validation
    )

    # i am gonna keep this result too so later i can compare all the months together
    vk_cv_results.append(
        vk_neural_fold_metrics
    )


# stage 5 - checking the temporal cross validation results

vk_print_heading(
    "STAGE 5 - TEMPORAL CROSS VALIDATION RESULTS"
)

vk_cv_results_data = pd.DataFrame(
    vk_cv_results
)

# lets arrange the columns so the results are easier for me to read
vk_cv_results_data = vk_cv_results_data[
    [
        "Validation_period",
        "Model",
        "Training_rows",
        "Validation_rows",
        "MAE_MW",
        "RMSE_MW",
        "R2",
        "MAPE_percent"
    ]
]

print(
    vk_cv_results_data
    .round(3)
    .to_string(
        index=False
    )
)

# now i am gonna calculate the average result and how much mae changed across the folds
vk_cv_summary_data = (
    vk_cv_results_data
    .groupby(
        "Model"
    )
    .agg(
        vk_mean_mae=(
            "MAE_MW",
            "mean"
        ),
        vk_std_mae=(
            "MAE_MW",
            "std"
        ),
        vk_mean_rmse=(
            "RMSE_MW",
            "mean"
        ),
        vk_mean_r2=(
            "R2",
            "mean"
        ),
        vk_mean_mape=(
            "MAPE_percent",
            "mean"
        )
    )
    .reset_index()
)

print(
    "\nsummary across validation periods:"
)

print(
    vk_cv_summary_data
    .round(3)
    .to_string(
        index=False
    )
)

# lets save both outputs because i would need them later when i am checking the results
vk_cv_results_data.to_csv(
    vk_task2_output_folder
    / "task2_01_temporal_cv_results.csv",
    index=False
)

vk_cv_summary_data.to_csv(
    vk_task2_output_folder
    / "task2_02_temporal_cv_summary.csv",
    index=False
)


# stage 6 - creating temporal cross validation graph

vk_print_heading(
    "STAGE 6 - CREATE TEMPORAL CROSS VALIDATION GRAPH"
)

# now i am gonna put the mae values into a shape that is easier to plot
vk_cv_plot_data = (
    vk_cv_results_data
    .pivot(
        index="Validation_period",
        columns="Model",
        values="MAE_MW"
    )
)

# lets keep the months in the actual chronological order
vk_cv_plot_data = (
    vk_cv_plot_data
    .reindex(
        [
            "July",
            "August",
            "September",
            "October"
        ]
    )
)

vk_cv_axis = (
    vk_cv_plot_data
    .plot(
        marker="o",
        figsize=(8, 5)
    )
)

vk_cv_axis.set_title(
    "Temporal Cross Validation MAE"
)

vk_cv_axis.set_xlabel(
    "Validation period"
)

vk_cv_axis.set_ylabel(
    "MAE (MW)"
)

plt.tight_layout()

plt.savefig(
    vk_task2_output_folder
    / "task2_03_temporal_cv_mae.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# stage 7 - setting up the learning curve

vk_print_heading(
    "STAGE 7 - LEARNING CURVE SETUP"
)

# now i want to check what happens when i give different amount of training data to the models
# i am gonna keep october fixed for validation so the comparison would be fair for every training size
vk_learning_validation_mask = (
    (
        vk_model_data[
            "vk_date"
        ]
        >= pd.Timestamp(
            "2014-10-01"
        )
    )
    &
    (
        vk_model_data[
            "vk_date"
        ]
        < pd.Timestamp(
            "2014-11-01"
        )
    )
)

# everything before october can potentially be used for training
vk_learning_training_pool = (
    vk_model_data[
        vk_model_data[
            "vk_date"
        ]
        < pd.Timestamp(
            "2014-10-01"
        )
    ]
    .copy()
)

vk_x_learning_validation = (
    vk_model_data
    .loc[
        vk_learning_validation_mask,
        vk_predictor_names
    ]
)

vk_y_learning_validation = (
    vk_model_data
    .loc[
        vk_learning_validation_mask,
        vk_target_name
    ]
)

# i am gonna test five different amounts of treining data
vk_training_fractions = [
    0.20,
    0.40,
    0.60,
    0.80,
    1.00
]

vk_learning_results = []


# stage 8 - running the learning curve experiment

vk_print_heading(
    "STAGE 8 - RUN LEARNING CURVE EXPERIMENT"
)

for vk_training_fraction in vk_training_fractions:

    # lets calculate how many observations belong to this training size
    vk_training_rows = int(
        len(
            vk_learning_training_pool
        )
        * vk_training_fraction
    )

    # now i am gonna take this amount of observations from the period before october
    # october would stay exactly the same every time so i can properly compare the different training sizes
    vk_learning_subset = (
        vk_learning_training_pool
        .iloc[
            -vk_training_rows:
        ]
        .copy()
    )

    vk_x_learning_train = (
        vk_learning_subset[
            vk_predictor_names
        ]
    )

    vk_y_learning_train = (
        vk_learning_subset[
            vk_target_name
        ]
    )

    print(
        "\ntraining fraction:",
        int(
            vk_training_fraction
            * 100
        ),
        "%"
    )

    print(
        "training observations:",
        vk_training_rows
    )

    # now i am gonna train the decision tree using this amount of data
    vk_learning_tree_model = DecisionTreeRegressor(
        max_depth=15,
        min_samples_leaf=100,
        random_state=42
    )

    vk_learning_tree_model.fit(
        vk_x_learning_train,
        vk_y_learning_train
    )

    # lets first see how well it fits the same training observations
    vk_learning_tree_train_prediction = (
        vk_learning_tree_model
        .predict(
            vk_x_learning_train
        )
    )

    # now lets predict the same october validation period
    vk_learning_tree_validation_prediction = (
        vk_learning_tree_model
        .predict(
            vk_x_learning_validation
        )
    )

    vk_tree_training_mae = (
        mean_absolute_error(
            vk_y_learning_train,
            vk_learning_tree_train_prediction
        )
    )

    vk_tree_validation_mae = (
        mean_absolute_error(
            vk_y_learning_validation,
            vk_learning_tree_validation_prediction
        )
    )

    vk_learning_results.append(
        {
            "Training_fraction":
                vk_training_fraction,
            "Training_percent":
                int(
                    vk_training_fraction
                    * 100
                ),
            "Training_rows":
                vk_training_rows,
            "Model":
                "Decision Tree",
            "Training_MAE_MW":
                vk_tree_training_mae,
            "Validation_MAE_MW":
                vk_tree_validation_mae
        }
    )

    # now i am gonna repeat the same experiment using the neural network
    vk_learning_x_scaler = StandardScaler()
    vk_learning_y_scaler = StandardScaler()

    # lets fit the scalers only on this training subset
    vk_learning_x_scaler.fit(
        vk_x_learning_train
    )

    vk_learning_y_scaler.fit(
        vk_y_learning_train
        .to_numpy()
        .reshape(-1, 1)
    )

    vk_learning_x_train_scaled = (
        vk_learning_x_scaler
        .transform(
            vk_x_learning_train
        )
    )

    # now i am gonna use the same scaler on october validation data
    vk_learning_x_validation_scaled = (
        vk_learning_x_scaler
        .transform(
            vk_x_learning_validation
        )
    )

    vk_learning_y_train_scaled = (
        vk_learning_y_scaler
        .transform(
            vk_y_learning_train
            .to_numpy()
            .reshape(-1, 1)
        )
        .ravel()
    )

    vk_learning_neural_model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.1,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    )

    vk_learning_neural_model.fit(
        vk_learning_x_train_scaled,
        vk_learning_y_train_scaled
    )

    # lets celculate predictions for the training observations first
    vk_learning_neural_train_scaled = (
        vk_learning_neural_model
        .predict(
            vk_learning_x_train_scaled
        )
    )

    vk_learning_neural_train_prediction = (
        vk_learning_y_scaler
        .inverse_transform(
            vk_learning_neural_train_scaled
            .reshape(-1, 1)
        )
        .ravel()
    )

    # now i am gonna predict the fixed october validation period
    vk_learning_neural_validation_scaled = (
        vk_learning_neural_model
        .predict(
            vk_learning_x_validation_scaled
        )
    )

    vk_learning_neural_validation_prediction = (
        vk_learning_y_scaler
        .inverse_transform(
            vk_learning_neural_validation_scaled
            .reshape(-1, 1)
        )
        .ravel()
    )

    vk_neural_training_mae = (
        mean_absolute_error(
            vk_y_learning_train,
            vk_learning_neural_train_prediction
        )
    )

    vk_neural_validation_mae = (
        mean_absolute_error(
            vk_y_learning_validation,
            vk_learning_neural_validation_prediction
        )
    )

    vk_learning_results.append(
        {
            "Training_fraction":
                vk_training_fraction,
            "Training_percent":
                int(
                    vk_training_fraction
                    * 100
                ),
            "Training_rows":
                vk_training_rows,
            "Model":
                "Neural Network",
            "Training_MAE_MW":
                vk_neural_training_mae,
            "Validation_MAE_MW":
                vk_neural_validation_mae
        }
    )


# stage 9 - checking the learning curve results

vk_print_heading(
    "STAGE 9 - LEARNING CURVE RESULTS"
)

vk_learning_results_data = pd.DataFrame(
    vk_learning_results
)

print(
    vk_learning_results_data
    .round(3)
    .to_string(
        index=False
    )
)

vk_learning_results_data.to_csv(
    vk_task2_output_folder
    / "task2_04_learning_curve_results.csv",
    index=False
)


# stage 10 - creating the learning curve graph

vk_print_heading(
    "STAGE 10 - CREATE LEARNING CURVE GRAPH"
)

plt.figure(
    figsize=(9, 6)
)

# lets first separate the decision tree learning results
vk_tree_learning_data = (
    vk_learning_results_data[
        vk_learning_results_data[
            "Model"
        ]
        == "Decision Tree"
    ]
)

# now i am gonna plot both training and validation mae for decision tree
plt.plot(
    vk_tree_learning_data[
        "Training_rows"
    ],
    vk_tree_learning_data[
        "Training_MAE_MW"
    ],
    marker="o",
    label="Decision Tree - Training"
)

plt.plot(
    vk_tree_learning_data[
        "Training_rows"
    ],
    vk_tree_learning_data[
        "Validation_MAE_MW"
    ],
    marker="o",
    label="Decision Tree - Validation"
)

# now i am gonna separate the neural network results
vk_neural_learning_data = (
    vk_learning_results_data[
        vk_learning_results_data[
            "Model"
        ]
        == "Neural Network"
    ]
)

# lets add both training and validation mae for neural network
plt.plot(
    vk_neural_learning_data[
        "Training_rows"
    ],
    vk_neural_learning_data[
        "Training_MAE_MW"
    ],
    marker="o",
    label="Neural Network - Training"
)

plt.plot(
    vk_neural_learning_data[
        "Training_rows"
    ],
    vk_neural_learning_data[
        "Validation_MAE_MW"
    ],
    marker="o",
    label="Neural Network - Validation"
)

plt.title(
    "Learning Curve Using Increasing Training Data"
)

plt.xlabel(
    "Number of training observations"
)

plt.ylabel(
    "MAE (MW)"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    vk_task2_output_folder
    / "task2_05_learning_curve.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# stage 11 - training the final models again for checking group errors

vk_print_heading(
    "STAGE 11 - TRAIN FINAL MODELS FOR FAIRNESS ANALYSIS"
)

# january to october is gonna remain final training like task 1
vk_final_train_mask = (
    vk_model_data[
        "vk_date"
    ]
    < pd.Timestamp(
        "2014-11-01"
    )
)

# november and december are gonna remain the unsen test period
vk_test_mask = (
    vk_model_data[
        "vk_date"
    ]
    >= pd.Timestamp(
        "2014-11-01"
    )
)

vk_x_final_train = (
    vk_model_data
    .loc[
        vk_final_train_mask,
        vk_predictor_names
    ]
)

vk_y_final_train = (
    vk_model_data
    .loc[
        vk_final_train_mask,
        vk_target_name
    ]
)

vk_x_test = (
    vk_model_data
    .loc[
        vk_test_mask,
        vk_predictor_names
    ]
)

vk_y_test = (
    vk_model_data
    .loc[
        vk_test_mask,
        vk_target_name
    ]
)

# now i am gonna recreate the final task 1 decision tree
vk_final_tree_model = DecisionTreeRegressor(
    max_depth=15,
    min_samples_leaf=100,
    random_state=42
)

vk_final_tree_model.fit(
    vk_x_final_train,
    vk_y_final_train
)

vk_final_tree_prediction = (
    vk_final_tree_model
    .predict(
        vk_x_test
    )
)

# now i am gonna recreate the final task 1 neural network
(
    vk_final_neural_model,
    vk_final_x_scaler,
    vk_final_y_scaler,
    vk_final_neural_prediction
) = vk_train_neural_model(
    vk_x_final_train,
    vk_y_final_train,
    vk_x_test
)


# stage 12 - creating the groups for checking model bias

vk_print_heading(
    "STAGE 12 - CREATE GROUPS FOR FAIRNESS ANALYSIS"
)

# now i am gonna look deeper into the errors instead of only checking one overall mae
# i want to see if the model is making different amount of errors under different conditions
# here i would be checking weekday vs weekend and different temperature groups
vk_fairness_test_data = (
    vk_model_data
    .loc[
        vk_test_mask,
        [
            "vk_is_weekend",
            "vk_max_temp",
            "TOTALDEMAND"
        ]
    ]
    .copy()
)

# lets give weekdays and weekends easy names
vk_weekend_group = (
    vk_fairness_test_data[
        "vk_is_weekend"
    ]
    .map(
        {
            0: "weekday",
            1: "weekend"
        }
    )
)

vk_weekend_group.name = "Group"

# i used similar temperature groups in task 1
# now i am gonna use them to see if model errors change under different temperature conditions
vk_temperature_group = pd.cut(
    vk_fairness_test_data[
        "vk_max_temp"
    ],
    bins=[
        -np.inf,
        15,
        25,
        30,
        np.inf
    ],
    labels=[
        "<=15 C",
        "15-25 C",
        "25-30 C",
        ">30 C"
    ]
)

vk_temperature_group.name = "Group"

print(
    "weekday and weekend counts:"
)

print(
    vk_weekend_group
    .value_counts()
)

print(
    "\ntemperature group counts:"
)

print(
    vk_temperature_group
    .value_counts(
        sort=False
    )
)


# stage 13 - checking weekday and weekend errors using fairlearn

vk_print_heading(
    "STAGE 13 - FAIRLEARN ANALYSIS BY WEEKDAY AND WEEKEND"
)

# i am gonna compare mae and rmse for every group
# this would help me understand if one group is getting much larger prediction errors than another group
vk_fairness_metrics = {
    "MAE_MW":
        mean_absolute_error,
    "RMSE_MW":
        vk_rmse_metric
}

# now i am gonna create one metricframe for the decision tree
vk_tree_weekend_frame = MetricFrame(
    metrics=vk_fairness_metrics,
    y_true=vk_y_test,
    y_pred=vk_final_tree_prediction,
    sensitive_features=vk_weekend_group
)

# now doing the same thing for the neural network
vk_neural_weekend_frame = MetricFrame(
    metrics=vk_fairness_metrics,
    y_true=vk_y_test,
    y_pred=vk_final_neural_prediction,
    sensitive_features=vk_weekend_group
)

vk_tree_weekend_results = (
    vk_tree_weekend_frame
    .by_group
    .reset_index()
)

vk_tree_weekend_results[
    "Model"
] = "Decision Tree"

vk_neural_weekend_results = (
    vk_neural_weekend_frame
    .by_group
    .reset_index()
)

vk_neural_weekend_results[
    "Model"
] = "Neural Network"

vk_weekend_fairness_results = pd.concat(
    [
        vk_tree_weekend_results,
        vk_neural_weekend_results
    ],
    ignore_index=True
)

print(
    vk_weekend_fairness_results
    .round(3)
    .to_string(
        index=False
    )
)

vk_weekend_fairness_results.to_csv(
    vk_task2_output_folder
    / "task2_06_weekend_fairness.csv",
    index=False
)


# stage 14 - checking temperature group errors using fairlearn

vk_print_heading(
    "STAGE 14 - FAIRLEARN ANALYSIS BY TEMPERATURE GROUP"
)

# now i am gonna repeat the group comparison for different temperature ranges
vk_tree_temperature_frame = MetricFrame(
    metrics=vk_fairness_metrics,
    y_true=vk_y_test,
    y_pred=vk_final_tree_prediction,
    sensitive_features=vk_temperature_group
)

vk_neural_temperature_frame = MetricFrame(
    metrics=vk_fairness_metrics,
    y_true=vk_y_test,
    y_pred=vk_final_neural_prediction,
    sensitive_features=vk_temperature_group
)

vk_tree_temperature_results = (
    vk_tree_temperature_frame
    .by_group
    .reset_index()
)

vk_tree_temperature_results[
    "Model"
] = "Decision Tree"

vk_neural_temperature_results = (
    vk_neural_temperature_frame
    .by_group
    .reset_index()
)

vk_neural_temperature_results[
    "Model"
] = "Neural Network"

vk_temperature_fairness_results = pd.concat(
    [
        vk_tree_temperature_results,
        vk_neural_temperature_results
    ],
    ignore_index=True
)

print(
    vk_temperature_fairness_results
    .round(3)
    .to_string(
        index=False
    )
)

vk_temperature_fairness_results.to_csv(
    vk_task2_output_folder
    / "task2_07_temperature_fairness.csv",
    index=False
)


# stage 15 - checking the error gaps between groups

vk_print_heading(
    "STAGE 15 - CALCULATE ERROR GAPS BETWEEN GROUPS"
)

vk_fairness_gap_rows = []

# now i am gonna find the gap between the group having highest mae and the group having lowest mae
# a bigger gap would tell me that the model is not performing equally across these conditions
for vk_model_name in [
    "Decision Tree",
    "Neural Network"
]:

    vk_model_weekend_data = (
        vk_weekend_fairness_results[
            vk_weekend_fairness_results[
                "Model"
            ]
            == vk_model_name
        ]
    )

    vk_weekend_mae_gap = (
        vk_model_weekend_data[
            "MAE_MW"
        ].max()
        -
        vk_model_weekend_data[
            "MAE_MW"
        ].min()
    )

    vk_fairness_gap_rows.append(
        {
            "Model":
                vk_model_name,
            "Grouping":
                "weekday vs weekend",
            "MAE_gap_MW":
                vk_weekend_mae_gap
        }
    )

    vk_model_temperature_data = (
        vk_temperature_fairness_results[
            vk_temperature_fairness_results[
                "Model"
            ]
            == vk_model_name
        ]
    )

    vk_temperature_mae_gap = (
        vk_model_temperature_data[
            "MAE_MW"
        ].max()
        -
        vk_model_temperature_data[
            "MAE_MW"
        ].min()
    )

    vk_fairness_gap_rows.append(
        {
            "Model":
                vk_model_name,
            "Grouping":
                "temperature groups",
            "MAE_gap_MW":
                vk_temperature_mae_gap
        }
    )


vk_fairness_gap_data = pd.DataFrame(
    vk_fairness_gap_rows
)

print(
    vk_fairness_gap_data
    .round(3)
    .to_string(
        index=False
    )
)

vk_fairness_gap_data.to_csv(
    vk_task2_output_folder
    / "task2_08_fairness_error_gaps.csv",
    index=False
)


# stage 16 - printing everything together at the end

vk_print_heading(
    "STAGE 16 - FINAL TASK 2 ANALYSIS SUMMARY"
)

# now that all the experiments are completed i am gonna print the important results together
# this is gonna make it easier for me to check the numbers before using them in my analysis
print(
    "\ntemporal cross validation summary:"
)

print(
    vk_cv_summary_data
    .round(3)
    .to_string(
        index=False
    )
)

print(
    "\nlearning curve results:"
)

print(
    vk_learning_results_data
    .round(3)
    .to_string(
        index=False
    )
)

print(
    "\nweekday and weekend fairness results:"
)

print(
    vk_weekend_fairness_results
    .round(3)
    .to_string(
        index=False
    )
)

print(
    "\ntemperature fairness results:"
)

print(
    vk_temperature_fairness_results
    .round(3)
    .to_string(
        index=False
    )
)

print(
    "\nfairness mae gaps:"
)

print(
    vk_fairness_gap_data
    .round(3)
    .to_string(
        index=False
    )
)

print(
    "\nall task 2 outputs have been saved to:"
)

print(
    vk_task2_output_folder.resolve()
)

print(
    "\ntask 2 analysis completed successfully."
)
