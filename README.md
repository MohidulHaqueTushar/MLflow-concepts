
# MLflow Tutorial: Manage the Complete Machine Learning Lifecycle

This tutorial provides a step-by-step guide to using MLflow—a powerful open-source platform to manage the Machine Learning lifecycle.

---

## Table of Contents

1. [What is MLflow?](#what-is-mlflow)
2. [Challenges in the Machine Learning Lifecycle](#challenges-in-the-machine-learning-lifecycle)
3. [MLflow Components](#mlflow-components)
    - [MLflow Tracking](#mlflow-tracking)
    - [MLflow Models](#mlflow-models)
    - [MLflow Model Registry](#mlflow-model-registry)
    - [MLflow Projects](#mlflow-projects)
4. [Hands-on Examples](#hands-on-examples)
5. [Model Deployment with MLflow](#model-deployment-with-mlflow)
6. [Workflows and Multi-step Pipelines](#workflows-and-multi-step-pipelines)
7. [Wrap-Up](#wrap-up)

---

## What is MLflow?

**MLflow** is a platform that helps manage the machine learning lifecycle including experimentation, reproducibility, and deployment. It integrates with many ML libraries and can be used with any language or tool.

---

## Challenges in the Machine Learning Lifecycle

Machine learning projects are complex. They require:
- Tracking many experiments
- Managing versions of models
- Collaborating across teams
- Deploying models reliably

MLflow helps unify these steps under one platform.

---

## MLflow Components

### MLflow Tracking

This component is used to record experiments including parameters, metrics, and artifacts (e.g., model files, data files).

```python
# Track parameters, metrics, and artifacts for reproducibility
import mlflow

mlflow.set_experiment("My Experiment")

with mlflow.start_run():
    mlflow.log_param("alpha", 0.1)
    mlflow.log_metric("rmse", 0.25)
    mlflow.log_artifact("train.py")  # Save training script as artifact
```

Start the UI to visually inspect runs:

```bash
mlflow ui
# Open browser at http://localhost:5000
```

---

### MLflow Models

MLflow Models packages models using standardized formats called "Flavors", allowing you to load and deploy them easily.

```python
# Train and log a model using sklearn flavor
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

mlflow.sklearn.log_model(model, "model")  # Save the model
```

Enable automatic logging:

```python
mlflow.sklearn.autolog()  # Automatically logs parameters, metrics, model
model.fit(X_train, y_train)
```

---

### MLflow Model Registry

This provides a central hub to manage versions of your ML models and control their promotion through stages (e.g., Staging → Production).

```python
# Register and transition a model to staging
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.create_registered_model("MyModel")

client.transition_model_version_stage(
    name="MyModel", version=1, stage="Staging"
)
```

---

### MLflow Projects

Projects package your ML code and environment for reproducibility.

```yaml
# MLproject file
name: salary_predictor

entry_points:
  main:
    parameters:
      alpha: {type: float, default: 0.1}
    command: "python train.py --alpha {alpha}"
```

Run the project locally or from a GitHub repo:

```bash
mlflow run . -P alpha=0.1
mlflow run https://github.com/user/repo -P alpha=0.1
```

---

## Hands-on Examples

### Track an Experiment

```python
# Log parameters and metrics
mlflow.set_experiment("Demo Experiment")

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.92)
```

### Query Experiments with Pandas

```python
# Load run history and filter results
import mlflow
import pandas as pd

runs_df = mlflow.search_runs(experiment_names=["Demo Experiment"])
print(runs_df.head())
```

---

## Model Deployment with MLflow

MLflow allows you to deploy models as REST APIs for real-time inference.

```bash
# Serve the model from a previous run
mlflow models serve -m runs:/<run_id>/model --port 5000
```

Use `curl` to send input data:

```bash
# Send a prediction request
curl -X POST http://localhost:5000/invocations   -H "Content-Type: application/json"   -d '{
        "columns": ["feature1", "feature2"],
        "data": [[1.5, 3.2]]
      }'
```

---

## Workflows and Multi-step Pipelines

Projects allow chaining multiple scripts in a single workflow.

```python
# Step-by-step execution with intermediate results
import mlflow.projects

step1 = mlflow.projects.run(".", entry_point="data_prep")
step2 = mlflow.projects.run(".", entry_point="train_model",
    parameters={"input_run_id": step1.run_id})
```

---

## Wrap-Up

MLflow simplifies many aspects of the ML development process:

- Tracking with MLflow Tracking
- Packaging with MLflow Models
- Versioning with the Model Registry
- Reproducibility with MLflow Projects
- Deployment through REST APIs

---

## Resources

- [MLflow Documentation](https://mlflow.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Databricks MLflow Guide](https://www.databricks.com/product/mlflow)
- [SHAP Explainability](https://shap.readthedocs.io/en/latest/index.html)

---

## License

This guide is for educational purposes. Code samples are simplified for clarity.

---

## Wrapup

We’ve learned how to implement, track, register, and deploy models using MLflow!