#!/bin/bash

if [[ -f mlflow.pid ]]; then
    kill $(cat logs/mlflow.pid) && rm logs/mlflow.pid
    echo "Stopped MLflow UI."
fi

if [[ -f prefect.pid ]]; then
    kill $(cat logs/prefect.pid) && rm logs/prefect.pid
    echo "Stopped Prefect UI."
fi