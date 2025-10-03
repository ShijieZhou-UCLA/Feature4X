# SSH Compatibility Guide for SAM2 Interactive Segmentation

## Overview

The SAM2 Interactive Segmentation tool is designed to work seamlessly over SSH connections, whether you have display forwarding enabled or not. Here's how it adapts to different scenarios:

## SSH Connection Scenarios

### Scenario 1: SSH with X11 Forwarding ✅
**Connection:** `ssh -X username@server` or `ssh -Y username@server`
- **Display:** Pop-up windows show reference images immediately
- **Interaction:** Standard GUI workflow
- **Best for:** When you have reliable, fast network connection

### Scenario 2: SSH without Display (Recommended) 📁
**Connection:** `ssh username@server`
- **Display:** Images automatically saved to files
- **Interaction:** Text-based coordinate input with saved reference images
- **Best for:** Most SSH scenarios, slower connections, headless servers

## How It Works

### Automatic Detection
The tool automatically detects your environment:

```python
# Automatically detects display availability
if 'DISPLAY' in os.environ:
    # Use interactive GUI mode
    plt.show()
else:
    # Use file-saving mode
    matplotlib.use('Agg')  # Non-interactive backend
    plt.savefig('reference_image.png')
```

### SSH Mode Features
When no display is detected:
- ⚠️  Warning message: "No display detected (SSH mode)"
- 📁 Reference images saved to output directory
- 📷 Clear file paths provided for easy copying
- 🔧 All functionality preserved - just different UI

## Usage Examples

### Interactive Mode (with display)
```bash
# Regular interactive mode
python sam2_segmentation.py --rendered_results_path /path/to/results.pth --save_dir /output

# Shows popup window with reference image
# User can see image immediately while entering coordinates
```

### SSH File Mode (no display)
```bash
# Same command, different behavior
python sam2_segmentation.py --rendered_results_path /path/to/results.pth --save_dir /output

# Output:
# ⚠️  No display detected (SSH mode). Using non-interactive backend.
# 📁 Reference images will be saved to files instead of displayed.
# 📷 Reference image saved to: /output/reference_frame_for_selection.png
# You can view it with: scp /output/reference_frame_for_selection.png your_local_machine:/path/
```

## Copying Images from Server

### Using SCP (Secure Copy)
```bash
# Copy single image
scp username@server:/path/to/reference_image.png ./

# Copy entire output directory
scp -r username@server:/path/to/output_directory ./
```

### Using rsync (for directories)
```bash
rsync -avz username@server:/path/to/output_directory/ ./local_directory/
```

## Best Practices for SSH Users

### 1. Use Descriptive Output Directories
```bash
python sam2_segmentation.py \
  --rendered_results_path /path/to/results.pth \
  --save_dir /output/sam2_$(date +%Y%m%d_%H%M%S)
```

### 2. Use tmux/screen for Long Sessions
```bash
# Start persistent session
tmux new-session -d -s sam2_session

# Attach to session
tmux attach-session -t sam2_session

# Run your segmentation
python sam2_segmentation.py --rendered_results_path ... --save_dir ...

# Detach (Ctrl+b, then d) and reconnect later
```

### 3. Pre-stage Reference Images
```bash
# Show reference frame first
python sam2_segmentation.py \
  --rendered_results_path /path/to/results.pth \
  --save_dir /output \
  --show_frame

# Copy image to view locally, then run with coordinates
scp username@server:/output/reference_frame.png ./
python sam2_segmentation.py \
  --rendered_results_path /path/to/results.pth \
  --save_dir /output \
  --point 200 150
```

## Troubleshooting

### Display Issues
```bash
# Check if display is available
echo $DISPLAY

# If empty, you're in SSH mode (file-based workflow)
# If shows ":10.0" or similar, X11 forwarding is working
```

### X11 Forwarding Problems
```bash
# Try trusted X11 forwarding
ssh -Y username@server

# Check X11 forwarding is enabled on server
# In /etc/ssh/sshd_config:
# X11Forwarding yes
# X11DisplayOffset 10
```

### File Permission Issues
```bash
# Ensure output directory is writable
mkdir -p /path/to/output
chmod 755 /path/to/output
```

## Performance Considerations

- **X11 Forwarding:** Slower over high-latency connections
- **File Mode:** Faster, works on any SSH connection
- **Network Usage:** File mode uses less bandwidth
- **Reliability:** File mode more reliable for unstable connections

## Summary

The SAM2 Interactive Segmentation tool automatically adapts to your SSH environment:
- **With display:** Traditional GUI workflow
- **Without display:** File-based workflow with same functionality
- **Always works:** No matter your SSH setup
- **User-friendly:** Clear instructions for each mode

Choose the method that works best for your network conditions and server setup!
