# Machine Learning Engineer Technical Assessment

Reference Dataset: [Madrid Housing Market](https://www.kaggle.com/code/smadler92/madrid-housing-market-machine-learning-model)

## Installing

Python 3.10.6 and Windows 11 were used to complete the test task.

Creation of virtual environment:

```python -m venv .venv```

Activation:

```.\.venv\Scripts\activate```

Installing the required packages:

```pip install -r requirements.txt```

& installing the required packages for dev:

```pip install -r requirements-dev.txt```

&

```pre-commit install```

And after that you need to use these commands:

```
pre-commit clean
pre-commit install
pre-commit run --all-files
```

## Usage

### Step 1: Notebook

After downloading the [dataset](https://www.kaggle.com/code/smadler92/madrid-housing-market-machine-learning-model), you need to use a notebook (```notebooks/01_eda.ipynb```) to analyze it.

In it, you load a dataset, get data about this dataset, draw histograms for the main positions, and get the correlation of the main values.

<img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/89d4adb6-3485-4f4c-8099-0ee94a2704a7" />
<img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/04e9e572-734e-4fc9-ab99-18e58731e102" />
<img width="1265" height="686" alt="image" src="https://github.com/user-attachments/assets/f8a4b427-0f2c-4ad0-a49d-4f6cfc449608" />

This will help you to use not all columns when preprocessing data, but those that have the greatest impact on the target.

### Step 2: Preprocess

This file (```src/preprocessing/preprocessing.py```) prepares raw real estate data for training machine learning models, converting it into a format suitable for price prediction algorithms.
The file can be used as an import module or run directly to test the pipeline on processed data. 
Run with the command ```python src/preprocessing/preprocessing.py``` and it will create ```models/preprocessor.joblib```, which we will then include in ```train.py```.

### Step 3: Split data

This file ```src/utils/split_data.py``` run with the command ```python src/utils/split_data.py```

Ensures reproducible data splitting so that:
* Models are evaluated on the same test data
* Cross-validation is consistent across experiments
* Avoid data leakage between train/test

Train/Test Split - splits the training and test sets:
* 80% of the data - training (default)
* 20% of the data - testing (default)
* Fixes the indices of the test set

K-Fold Cross-Validation - creates a 5-fold cross-validation:
* Splits the training set into 5 folds
* Determines the train/validation indices for each fold

The ```data/splits/``` folder will contain:

```test_indices.json``` → a list of indices for the test,

```cv_folds.json``` → a dictionary with train/val indices for each fold.

### Step 4: Train

This file ```src/models/train.py``` run with the command ```python -m src.models.train```

This file is the main training script for training models.

1. Cross-validation model training:
* Linear Regression (base model)
* LightGBM with hyperparameter selection via GridSearchCV

2. Performance evaluation:
* Calculates metrics: MAE, RMSE, R²
* Conducts evaluation on each cross-validation fold
* Aggregates results across all folds

3. Final training:
* Trains the best model on all training data
* Saves the final model
* Evaluates on test data

```models/final_model.joblib``` - the final trained model
```models/metrics.json``` - all metrics and parameters

### Step 5: Tracking experiments

This file ```src/models/train_mlflow.py``` you may run with the command ```python -m src.models.train_mlflow --config config/experiments/exp1.yaml``` for 1 experiment,
or the file ```src/models/run_experiments.py``` run with the command 
```python src/models/run_experiments.py --configs config/experiments/exp1.yaml config/experiments/exp2.yaml config/experiments/exp3.yaml config/experiments/exp4.yaml``` for all 4 experiments.

After running the 4 experiments:

MLflow dashboard:

Open http://127.0.0.1:5000 in browser.

Use the web UI to select the experiment madrid_housing_experiments. You should see the four runs.

Screenshots:

<img width="981" height="142" alt="image" src="https://github.com/user-attachments/assets/53edcae6-5ce2-4b78-92e6-8c43d7e6711f" />

<img width="939" height="143" alt="image" src="https://github.com/user-attachments/assets/52a1f8a9-66e4-497b-b753-e00db277b116" />

<img width="932" height="138" alt="image" src="https://github.com/user-attachments/assets/aefe9680-6165-4e68-84ce-375c5ce69a80" />

<img width="893" height="146" alt="image" src="https://github.com/user-attachments/assets/14f7660c-a946-4d49-9915-d3c80ff5032e" />

### Step 6: Evaluation

This file ```src/models/evaluate_all.py``` run with the command ```python -m src.models.evaluate_all```

A separate folder is created for each model in the reports/ folder:
Quality metrics:

```metrics.json``` - MAE, RMSE, R², log-MAE

Visualizations:

```pred_vs_actual.png``` - Predicted vs. Actual values ​​plot

```residuals.png``` - Residuals (errors) histogram

Feature importance analysis:

```feature_importances.csv``` - Feature importance table

```feature_importances.png``` - Feature importance horizontal plot

```shap_summary.png``` - SHAP summary plot (if available)

Summary report:

```summary_metrics.csv``` - Comparison of metrics for all models

### Step 7: CLI & Makefile

**CLI**

This file (```src/cli.py```) is the command-line interface (CLI) for the entire Madrid house price forecasting project. It provides a single entry point for all project operations. 
You can run it with ```python -m src.cli prepare-data [--config path/to/config.yaml]``` to prepare data, 
* ```python -m src.cli train [--config path/to/config.yaml]``` to train models,
* ```python -m src.cli evaluate [--config] [--all]``` to evaluate models,
* ```python -m src.cli serve [--config path/to/config.yaml]``` to serve models,
* ```python -m src.cli run-experiments --configs config/experiments/exp1.yaml --configs config/experiments/exp2.yaml ...``` to run experiments.

**Makefile** (for Linux)

Commands:

* ```make install``` To install requirements
* ```make prepare-data``` To prepare data
* ```make train``` To train model
* ```make evaluate``` To evaluate models
* ```make serve``` To serve
* ```make run-experiments``` To run experiments
* ```make clean``` To clean cash
* ```make lint``` To check code with ```flake8```
* ```make test``` To run tests

### Step 8: App

Run it with ```uvicorn src.app.main:app --reload```

```src/app/main.py```

The main API module.

What it does:
* Configures FastAPI (name, version, description).
* Loads the ML model at startup (via load_model() from utils.py).
* Creates endpoints for the API.

Endpoints:

```GET /health```

Checks if the model is available.
* ```{"status": "ok"}``` if the model is loaded.
* ```{"status": "error"}``` if not.

```GET /model/info```

Returns meta information about the model

```POST /predict```

Makes a prediction for one object.

For example:
input:
```
{
  "sq_mt_built": 80,
  "sq_mt_useful": 70,
  "n_rooms": 3,
  "n_bathrooms": 2,
  "has_parking": true
}
```

Output:
```
{
  "prediction": 325000.0
}
```

```POST /batch_predict```

Makes predictions for multiple objects at once.

For example:
input:
```
{
  "inputs": [
    {"sq_mt_built": 60, "sq_mt_useful": 55, "n_rooms": 2, "n_bathrooms": 1, "has_parking": false},
    {"sq_mt_built": 120, "sq_mt_useful": 110, "n_rooms": 4, "n_bathrooms": 2, "has_parking": true}
  ]
}
```

Output:
```
{
  "predictions": [210000.0, 520000.0]
}
```

Full commands:

* ```curl -X GET "http://127.0.0.1:8000/health" -H "accept: application/json"``` or ```Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method GET```
* ```curl -X GET "http://127.0.0.1:8000/model/info" -H "accept: application/json"``` or ```Invoke-RestMethod -Uri "http://127.0.0.1:8000/model/info" -Method GET```
* ```
  curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
  "sq_mt_built": 80,
  "sq_mt_useful": 70,
  "n_rooms": 3,
  "n_bathrooms": 2,
  "has_parking": true
  }'
  ```

  or
  
  ```
  $body = @{
      sq_mt_built = 80
      sq_mt_useful = 70
      n_rooms = 3
      n_bathrooms = 2
      has_parking = $true
  } | ConvertTo-Json
  
  Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method POST -Body $body -ContentType "application/json"
  ```
* ```
  curl -X POST "http://127.0.0.1:8000/batch_predict" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "sq_mt_built": 60,
        "sq_mt_useful": 55,
        "n_rooms": 2,
        "n_bathrooms": 1,
        "has_parking": false
      },
      {
        "sq_mt_built": 120,
        "sq_mt_useful": 110,
        "n_rooms": 4,
        "n_bathrooms": 2,
        "has_parking": true
      }
    ]
  }'
  ```
  or
  ```
  $body = @{
    inputs = @(
        @{
            sq_mt_built = 60
            sq_mt_useful = 55
            n_rooms = 2
            n_bathrooms = 1
            has_parking = $false
        },
        @{
            sq_mt_built = 120
            sq_mt_useful = 110
            n_rooms = 4
            n_bathrooms = 2
            has_parking = $true
        }
    )
  } | ConvertTo-Json -Depth 3
  
  Invoke-RestMethod -Uri "http://127.0.0.1:8000/batch_predict" -Method POST -Body $body -ContentType "application/json"
  ```

### Step 9: Docker

You can also run it throught Docker using commands:

```docker build -t madrid-housing-api .```

```docker run -p 8000:8000 madrid-housing-api```

or using docker-compose

```docker-compose up --build```

And example commands for container:

```
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method GET
Invoke-RestMethod -Uri "http://127.0.0.1:8000/model/info" -Method GET
# batch predict
$body = @{
    inputs = @(
        @{ sq_mt_built=60; sq_mt_useful=55; n_rooms=2; n_bathrooms=1; has_parking=$false },
        @{ sq_mt_built=120; sq_mt_useful=110; n_rooms=4; n_bathrooms=2; has_parking=$true }
    )
} | ConvertTo-Json -Depth 3
Invoke-RestMethod -Uri "http://127.0.0.1:8000/batch_predict" -Method POST -Body $body -ContentType "application/json"
```

### Step 10: Tests

You can run tests using the command ```pytest --cov=src --cov-report=term-missing```. Tests cover **94%**.
