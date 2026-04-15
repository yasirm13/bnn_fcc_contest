#!/bin/bash

# 1. Store the current working directory
START_DIR=$(pwd)

# 2. Navigate to home and source the config
# We use 'source' so the exported variables stay in your current shell
cd ~ && source /apps/reconfig/enable_pro

# 3. Return to the original directory
cd "$START_DIR"

echo "ECE apps activated and returned to $START_DIR"