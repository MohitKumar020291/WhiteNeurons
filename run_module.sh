#!/bin/bash

echo "The script name is: $0"

# 
if [ "$1" = "main" ]; then
    echo "running main.py"
    python3 -m main
elif [ "$1" = "test_unet" ]; then
    python3 -m model.AutoAnnotate.UNet.test_unet
elif [ "$1" = "experiment_unet" ]; then
    python3 -m model.AutoAnnotate.UNet.experiment_unet
elif [ "$1" = "seg_unet" ]; then
    if [ -n "$2" ]; then
        echo "Passing update_cache=$2"
        echo "running model.AutoAnnotate.AutoAnnotate.seg_unet.py $2"
        python3 -m model.AutoAnnotate.AutoAnnotate.seg_unet update_cache="$2"
    else
        echo "No update_cache arg passed, running without it"
        echo "running model.AutoAnnotate.AutoAnnotate.seg_unet.py"
        python3 -m model.AutoAnnotate.AutoAnnotate.seg_unet
    fi
fi