#!/bin/bash
# Usage: ./run_benchmark_new.sh "Your commit message here"

set -e

COMMIT_MSG="$1"
if [ -z "$COMMIT_MSG" ]; then
  echo "Usage: $0 'Your commit message'"
  exit 1
fi

IMAGE=section4-inference-new
CONTAINER_NAME=section4-inference-new-bench
PORT=8080

# Stop any running container on the same port/name
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
  echo "Stopping existing container..."
  docker rm -f $CONTAINER_NAME
fi



# Ensure logs directory exists
LOGDIR="logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/server_logs_$(date +%Y%m%d_%H%M%S).log"


docker run --rm --gpus all -d --name $CONTAINER_NAME -p $PORT:8080 -v $(pwd)/new:/app $IMAGE

# Wait for the server to be ready
for i in {1..30}; do
  if curl -s http://localhost:$PORT/health | grep -q 'ok'; then
    echo "Server is up!"
    break
  fi
  echo "Waiting for server... ($i)"
  sleep 1
done

# Start log tailing in background
docker logs -f $CONTAINER_NAME > "$LOGFILE" 2>&1 &
LOG_PID=$!



# Run the benchmark
python3 benchmark_inference_server.py --commit-message "$COMMIT_MSG"

# Generate/update benchmark history plot
python3 plot_benchmark_history.py benchmark_history.csv benchmark_history.png

# Stop the container
docker rm -f $CONTAINER_NAME

# Stop log tailing
kill $LOG_PID
echo "Server logs saved to $LOGFILE"
