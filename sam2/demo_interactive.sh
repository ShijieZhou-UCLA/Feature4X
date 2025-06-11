#!/bin/bash

# Interactive SAM2 Segmentation Demo
# This script demonstrates the interactive terminal mode

echo "Starting Interactive SAM2 Segmentation Demo..."
echo ""

# First, run the interactive mode
echo "Step 1: Running interactive mode (shows image and waits for input)"
echo "You will see:"
echo "- Reference image displayed with coordinate grid"
echo "- Terminal prompt asking for selection mode"
echo "- Input prompts for coordinates"
echo ""

read -p "Press Enter to start the demo..."

cd /usr/project/Feature4X/sam2
conda run -n feature4x python sam2_segmentation.py \
    --rendered_results_path /usr/project/Feature4X/output/bear/final_viz/81_round_moving/rendered_results.pth \
    --save_dir ../demo_interactive_terminal

echo ""
echo "Demo completed!"
echo "Check the output directory: ../demo_interactive_terminal"
