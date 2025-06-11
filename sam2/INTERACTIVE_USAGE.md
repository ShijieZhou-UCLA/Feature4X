# Interactive SAM2 Segmentation Usage Guide

The SAM2 segmentation script now supports an interactive mode that helps users select points or bounding boxes by first showing the reference image with coordinates.

## Usage Options

### 1. Interactive Mode (Show Reference Image)
When you don't provide point or box coordinates, the script automatically shows a reference image with a grid to help you identify coordinates:

```bash
python sam2_segmentation.py --rendered_results_path /path/to/rendered_results.pth --save_dir /path/to/output
```

This will:
- Display the reference frame with coordinate grid
- Save a reference image (`reference_frame.png`) in your output directory
- Show you the exact command examples to run with coordinates

### 2. Show Frame Mode
Use `--show_frame` to display the reference image even when you provide coordinates:

```bash
python sam2_segmentation.py --rendered_results_path /path/to/rendered_results.pth --point 200 150 --save_dir /path/to/output --show_frame
```

### 3. Traditional Command Line Mode
Provide coordinates directly (as before):

**Point Selection:**
```bash
python sam2_segmentation.py --rendered_results_path /path/to/rendered_results.pth --point 200 150 --save_dir /path/to/output
```

**Box Selection:**
```bash
python sam2_segmentation.py --rendered_results_path /path/to/rendered_results.pth --box 100 100 300 300 --save_dir /path/to/output
```

**Combined Point and Box:**
```bash
python sam2_segmentation.py --rendered_results_path /path/to/rendered_results.pth --point 200 150 --box 100 100 300 300 --save_dir /path/to/output
```

## Interactive Workflow

1. **Run without coordinates** to see the reference image
2. **Note the coordinates** from the displayed image
3. **Run again with coordinates** to perform segmentation

### Example Workflow:

```bash
# Step 1: See the reference image
python sam2_segmentation.py --rendered_results_path /usr/project/Feature4X/output/bear/final_viz/81_round_moving/rendered_results.pth --save_dir ../try_sam_interactive

# Output will show:
# Reference frame: 0000.jpg
# Image size: 854x480 (width x height)
# Reference image saved to: ../try_sam_interactive/reference_frame.png
# 
# No point or box coordinates provided.
# Please run again with:
#   --point X Y                    (for point selection)
#   --box X1 Y1 X2 Y2             (for box selection)

# Step 2: Use the coordinates you identified
python sam2_segmentation.py --rendered_results_path /usr/project/Feature4X/output/bear/final_viz/81_round_moving/rendered_results.pth --point 200 150 --save_dir ../try_sam_interactive
```

## Features

- **Reference Image**: Automatically saved with coordinate grid for easy identification
- **Coordinate Guidance**: Shows image dimensions and provides exact command examples
- **Flexible Input**: Supports point-only, box-only, or combined selection
- **Visual Feedback**: Grid overlay helps identify precise coordinates
- **Backward Compatible**: All existing command-line usage continues to work

## File Outputs

The script generates:
- `reference_frame.png` - Reference image with coordinate grid
- `sam2_prompt.json` - Selected coordinates in JSON format
- `sam2_masks.pth` - Segmentation masks
- Visualization video showing segmentation results

## Tips

- Use the reference image to identify object locations precisely
- The coordinate system starts at (0,0) in the top-left corner
- For boxes, provide: x1 (left), y1 (top), x2 (right), y2 (bottom)
- The `--show_frame` flag is useful to verify your coordinate selection
