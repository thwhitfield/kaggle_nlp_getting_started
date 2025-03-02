#!/bin/bash

# Start MLflow UI
# nohup mlflow ui --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &
nohup mlflow ui > logs/mlflow.log 2>&1 &
MLFLOW_PID=$!

# Start Prefect UI
nohup prefect server start > logs/prefect.log 2>&1 &
PREFECT_PID=$!

echo "MLflow UI started with PID: $MLFLOW_PID (logs: logs/mlflow.log)"
echo "Prefect UI started with PID: $PREFECT_PID (logs: logs/prefect.log)"

# Save PIDs for later cleanup
echo $MLFLOW_PID > logs/mlflow.pid
echo $PREFECT_PID > logs/prefect.pid

# Wait a few seconds to ensure servers are up
sleep 5

# Open URLs in browser (Linux/macOS/Windows)
MLFLOW_URL="http://127.0.0.1:5000"
PREFECT_URL="http://127.0.0.1:4200"

if command -v xdg-open &> /dev/null; then
    xdg-open "$MLFLOW_URL"
    xdg-open "$PREFECT_URL"
elif command -v open &> /dev/null; then
    open "$MLFLOW_URL"
    open "$PREFECT_URL"
elif command -v start &> /dev/null; then
    start "$MLFLOW_URL"
    start "$PREFECT_URL"
else
    echo "Could not detect web browser command. Open manually:"
    echo "MLflow UI: $MLFLOW_URL"
    echo "Prefect UI: $PREFECT_URL"
fi