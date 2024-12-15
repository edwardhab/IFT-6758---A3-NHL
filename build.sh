#!/bin/bash

# Define image tag
IMAGE_TAG="ift6758/serving:latest"

# Build the Docker image
echo "Building the Docker image with tag: $IMAGE_TAG"
docker build -t $IMAGE_TAG -f Dockerfile.serving .

# Check if the build was successful
if [ $? -eq 0 ]; then
  echo "Docker image built successfully."
else
  echo "Failed to build the Docker image."
  exit 1
fi
