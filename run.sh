#!/bin/bash

# Load environment variables from the .env file
if [ -f .env ]; then
  export $(cat .env | xargs)
else
  echo ".env file not found!"
  exit 1
fi

# Check if the WANDB_API_KEY is loaded
if [ -z "$WANDB_API_KEY" ]; then
  echo "WANDB_API_KEY is not set in the .env file!"
  exit 1
fi

# Define container parameters
CONTAINER_NAME="ift-6758---a3-nhl"
IMAGE_TAG="ift6758/serving:latest"
HOST_PORT=5000
CONTAINER_PORT=5000

# Run the Docker container
echo "Running the Docker container with name: $CONTAINER_NAME"
docker run --rm -d \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -p $HOST_PORT:$CONTAINER_PORT \
  --name $CONTAINER_NAME \
  $IMAGE_TAG

# Check if the container started successfully
if [ $? -eq 0 ]; then
  echo "Docker container started successfully. Access the application at http://localhost:$HOST_PORT"
else
  echo "Failed to start the Docker container."
  exit 1
fi
