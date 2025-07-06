#!/bin/bash

echo "The script name is: $0"

# 
if [ "$1" = "main" ]; then
    echo "running main.py"
    python3 -m main
elif [ "$1" = "test_unet" ]; then
    python3 -m model.AutoAnnotate.UNet.test_unet
fi