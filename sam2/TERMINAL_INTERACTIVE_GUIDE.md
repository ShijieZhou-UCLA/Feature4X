# Terminal Interactive SAM2 Usage Guide

The SAM2 segmentation script now supports a single-run interactive terminal mode where you can see the reference image first, then input coordinates in the same session.

## How It Works

1. **Run without coordinates** - The script will:
   - Display the reference image with coordinate grid
   - Save a reference image file
   - Wait for your input in the terminal
   - Process your selection and run segmentation

2. **Interactive Terminal Session** - You'll see:
   - Image displayed (if GUI available) or saved to file
   - Menu to choose point or box selection
   - Prompts to enter coordinates
   - Validation of your input
   - Immediate segmentation processing

## Usage

### Simple Interactive Mode
```bash
python sam2_segmentation.py --rendered_results_path /path/to/rendered_results.pth --save_dir /output/dir
```

### With Frame Preview
```bash  
python sam2_segmentation.py --rendered_results_path /path/to/rendered_results.pth --point 200 150 --save_dir /output/dir --show_frame
```

## Interactive Workflow Example

```bash
cd /usr/project/Feature4X/sam2
conda run -n feature4x python sam2_segmentation.py \
    --rendered_results_path /usr/project/Feature4X/output/bear/final_viz/81_round_moving/rendered_results.pth \
    --save_dir ../my_segmentation_output
```

**What happens:**
1. Image loads and displays (854x480 resolution)
2. Reference image saved to: `../my_segmentation_output/reference_frame.png`
3. Terminal shows interactive menu:
   ```
   Choose selection mode:
   1. Point Selection (select one point)
   2. Box Selection (select bounding box) 
   3. Exit without selection
   
   Enter your choice (1, 2, or 3): 
   ```

**For Point Selection (choice 1):**
```
Enter X coordinate (horizontal): 200
Enter Y coordinate (vertical): 150
✓ Point selected: (200, 150)
```

**For Box Selection (choice 2):**
```
Enter X1 (left edge): 100
Enter Y1 (top edge): 100  
Enter X2 (right edge): 300
Enter Y2 (bottom edge): 300
✓ Box selected: [100, 100, 300, 300]
```

4. Segmentation processes automatically
5. Results saved to output directory

## Key Features

- **Single Run**: No need to run the script twice
- **Visual Reference**: Image displayed and saved for coordinate identification
- **Input Validation**: Checks coordinate bounds and validity
- **Error Recovery**: Invalid input prompts for retry
- **Clean Exit**: Option to exit without selection

## Output Files

- `reference_frame.png` - Reference image with coordinate grid
- `sam2_prompt.json` - Selected coordinates in JSON format  
- `sam2_masks.pth` - Segmentation masks
- Video visualization of results

## Tips

- The reference image is always saved even if display fails
- Use the saved reference image to identify coordinates if needed
- Coordinate system: (0,0) is top-left corner
- For boxes: x1,y1 = top-left, x2,y2 = bottom-right
- Press Ctrl+C to cancel at any time
