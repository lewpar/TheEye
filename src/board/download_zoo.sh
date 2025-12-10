#!/bin/bash

# Create base zoo path
mkdir -p ./zoo

# Setup hailo zoo Path
mkdir -p ./zoo/hailo

# Get HAILO Inferencing Models
echo "Downloading HAILO models.."

degirum download-zoo \
    --url "https://hub.degirum.com/degirum/hailo" \
    --path "./zoo/hailo"

echo "Finished downloading HAILO models."
