# Authors: Hui Ren (rhfeiyang.github.io)
#!/usr/bin/env python3
"""
Feature4X Interactive Interface
===============================

A unified interactive terminal interface for all Feature4X functionalities.
This script provides an easy-to-use menu system for accessing:
- SAM2 Segmentation
- Semantic Understanding  
- Language-guided Editing
- 4D Visual Question Answering (VQA)

Usage:
    python feature4x_interactive.py

Requirements:
    - conda environment 'feature4x' should be activated
    - Run from the               if no                 print(f"❌ No rendered results found for {data_name}!")
            return
            
        views = sorted(views)
        print(f"\n🎭 Available views for {data_name}:")
        for i, view in enumerate(views, 1):
            print(f"  {i}. {view}")
            
        # Select view
        while True: = sorted(views)
        print(f"\n🎭 Available views for {data_name}:")
        for i, view in enumerate(views, 1):
            print(f"  {i}. {view}")
            
        # Select view
        while True:iews = sorted(views)
        print(f"\n🎭 Available views for {data_name}:")
        for i, view in enumerate(views, 1):iews:
            print(f"❌ No rendered results found for {data_name}!")
            return
            
        views = sorted(views)
        print(f"\n🎭 Available views for {data_name}:")
        for i, view in enumerate(views, 1):
            print(f"  {i}. {view}")
            
        # Select viewviews = sorted(views)
        print(f"\n🎭 Available views for {data_name}:")
        for i, view in enumerate(views, 1):      print(f"\n🎭 Available views for {data_name}:")ws = sorted(views)
        print(f"\n🎭 Available views for {data_name}:")
        for i, view in enumerate(views, 1):oject root directory
"""

import os
import sys
import subprocess
import glob
from pathlib import Path
import argparse

class Feature4XInterface:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.ensure_project_root()
        
    def ensure_project_root(self):
        """Ensure we're in the correct project directory"""
        if not os.path.exists(os.path.join(self.project_root, "sam2")) or \
           not os.path.exists(os.path.join(self.project_root, "lseg_encoder")):
            print("❌ Error: This script must be run from the Feature4X project root directory!")
            print(f"Current directory: {self.project_root}")
            sys.exit(1)
            
    def check_conda_env(self):
        """Check if feature4x conda environment is activated"""
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
        if conda_env != 'feature4x':
            print("⚠️  Warning: 'feature4x' conda environment is not activated!")
            print("Please run: conda activate feature4x")
            response = input("Continue anyway? (y/N): ").lower().strip()
            if response != 'y':
                sys.exit(1)
        else:
            print("✅ conda environment 'feature4x' is active")
            
    def find_rendered_results(self):
        """Find available rendered results files in the output directory"""
        pattern = os.path.join(self.project_root, "output", "**", "rendered_results.pth")
        files = glob.glob(pattern, recursive=True)
        
        if not files:
            print("❌ No rendered_results.pth files found in output directory!")
            print("Please ensure you have processed data available.")
            return []
            
        # Convert to relative paths
        relative_files = []
        for file in files:
            rel_path = os.path.relpath(file, self.project_root)
            relative_files.append(rel_path)
            
        return sorted(relative_files)
        
    def run_command(self, command, description):
        """Run a command and handle errors"""
        print(f"\n🚀 {description}")
        print(f"Command: {' '.join(command)}")
        print("-" * 60)
        
        try:
            result = subprocess.run(command, cwd=self.project_root, check=True)
            print(f"\n✅ {description} completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {description} failed with exit code {e.returncode}")
            return False
        except KeyboardInterrupt:
            print(f"\n⏹️  {description} interrupted by user")
            return False
            
    def sam2_segmentation(self):
        """SAM2 Interactive Segmentation"""
        print("\n" + "="*60)
        print("🎯 SAM2 Interactive Segmentation")
        print("="*60)
        
        # Extract data names from available rendered results
        rendered_files = self.find_rendered_results()
        if not rendered_files:
            print("❌ No rendered results found!")
            return
            
        # Extract unique data names from paths like "output/bear/final_viz/*/rendered_results.pth"
        data_names = set()
        for file in rendered_files:
            if file.startswith("output/"):
                parts = file.split("/")
                if len(parts) >= 2:
                    data_names.add(parts[1])  # Extract "bear" from "output/bear/..."
                    
        if not data_names:
            print("❌ No valid data found in output directory!")
            return
            
        data_names = sorted(list(data_names))
        
        print("📂 Available datasets:")
        for i, name in enumerate(data_names, 1):
            print(f"  {i}. {name}")
            
        # Select dataset
        while True:
            try:
                choice = input(f"\nSelect dataset (1-{len(data_names)}): ").strip()
                choice = int(choice)
                if 1 <= choice <= len(data_names):
                    data_name = data_names[choice - 1]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(data_names)}")
            except ValueError:
                print("❌ Please enter a valid number")
                
        # Find available views for the selected dataset
        base_dir = os.path.join(self.project_root, "output", data_name, "final_viz")
        views = []
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                view_path = os.path.join(base_dir, item)
                if os.path.isdir(view_path) and os.path.exists(os.path.join(view_path, "rendered_results.pth")):
                    # Exclude 3D_moving for SAM2 compatibility
                    if "3D_moving" not in item:
                        views.append(item)
                    
        if not views:
            print(f"❌ No compatible rendered results found for {data_name}!")
            print("💡 Note: 3D_moving views are excluded for SAM2 compatibility")
            return
            
        views = sorted(views)
        print(f"\n🎭 Available views for {data_name}:")
        for i, view in enumerate(views, 1):
            print(f"  {i}. {view}")
            
        # Select view
        while True:
            try:
                choice = input(f"\nSelect view (1-{len(views)}): ").strip()
                choice = int(choice)
                if 1 <= choice <= len(views):
                    view = views[choice - 1]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(views)}")
            except ValueError:
                print("❌ Please enter a valid number")
                
        print(f"\n✅ Using dataset: {data_name}")
        print(f"✅ Using view: {view}")
        
        # Construct the rendered_results_path
        rendered_path = f"output/{data_name}/final_viz/{view}/rendered_results.pth"
        
        print("\nSegmentation options:")
        print("1. Interactive mode (select coordinates on image)")
        print("2. Specify point coordinates")
        print("3. Specify bounding box")
        print("4. Both point and bounding box")
        
        mode = input("\nSelect mode (1-4): ").strip()
        
        command = ["python", "sam2/sam2_segmentation.py", "--rendered_results_path", rendered_path]
        
        if mode == "2":
            x = input("Enter point X coordinate: ").strip()
            y = input("Enter point Y coordinate: ").strip()
            command.extend(["--point", x, y])
        elif mode == "3":
            x1 = input("Enter box X1 coordinate: ").strip()
            y1 = input("Enter box Y1 coordinate: ").strip()
            x2 = input("Enter box X2 coordinate: ").strip()
            y2 = input("Enter box Y2 coordinate: ").strip()
            command.extend(["--box", x1, y1, x2, y2])
        elif mode == "4":
            x = input("Enter point X coordinate: ").strip()
            y = input("Enter point Y coordinate: ").strip()
            x1 = input("Enter box X1 coordinate: ").strip()
            y1 = input("Enter box Y1 coordinate: ").strip()
            x2 = input("Enter box X2 coordinate: ").strip()
            y2 = input("Enter box Y2 coordinate: ").strip()
            command.extend(["--point", x, y, "--box", x1, y1, x2, y2])
            
        self.run_command(command, "SAM2 Segmentation")
        
    def semantic_understanding(self):
        """Semantic Understanding with LSeg"""
        print("\n" + "="*60)
        print("🧠 Semantic Understanding")
        print("="*60)
        
        # Extract data names from available rendered results
        rendered_files = self.find_rendered_results()
        if not rendered_files:
            print("❌ No rendered results found!")
            return
            
        # Extract unique data names from paths like "output/bear/final_viz/*/rendered_results.pth"
        data_names = set()
        for file in rendered_files:
            if file.startswith("output/"):
                parts = file.split("/")
                if len(parts) >= 2:
                    data_names.add(parts[1])  # Extract "bear" from "output/bear/..."
                    
        if not data_names:
            print("❌ No valid data found in output directory!")
            return
            
        data_names = sorted(list(data_names))
        
        print("📂 Available datasets:")
        for i, name in enumerate(data_names, 1):
            print(f"  {i}. {name}")
            
        # Select dataset
        while True:
            try:
                choice = input(f"\nSelect dataset (1-{len(data_names)}): ").strip()
                choice = int(choice)
                if 1 <= choice <= len(data_names):
                    data_name = data_names[choice - 1]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(data_names)}")
            except ValueError:
                print("❌ Please enter a valid number")
                
        # Find available views for the selected dataset
        base_dir = os.path.join(self.project_root, "output", data_name, "final_viz")
        views = []
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                view_path = os.path.join(base_dir, item)
                if os.path.isdir(view_path) and os.path.exists(os.path.join(view_path, "rendered_results.pth")):
                    views.append(item)
                    
        if not views:
            print(f"❌ No rendered results found for {data_name}!")
            return
            
        views = sorted(views)
        print(f"\n🎭 Available views for {data_name}:")
        for i, view in enumerate(views, 1):
            compatibility_note = " (Note: 3D_moving may have compatibility issues)" if "3D_moving" in view else ""
            print(f"  {i}. {view}{compatibility_note}")
            
        # Select view
        while True:
            try:
                choice = input(f"\nSelect view (1-{len(views)}): ").strip()
                choice = int(choice)
                if 1 <= choice <= len(views):
                    view = views[choice - 1]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(views)}")
            except ValueError:
                print("❌ Please enter a valid number")
                
        print(f"\n✅ Using dataset: {data_name}")
        print(f"✅ Using view: {view}")
        
        # Construct the rendered_results_path
        rendered_path = f"output/{data_name}/final_viz/{view}/rendered_results.pth"
        
        command = ["python", "lseg_encoder/lseg_inference.py", "--rendered_results_path", rendered_path]
        
        # Optional: custom labels
        use_custom_labels = input("\nUse custom labels? (y/N): ").lower().strip() == 'y'
        if use_custom_labels:
            labels = input("Enter labels (comma-separated): ").strip()
            command.extend(["--labels", labels])
            
        # Optional: save features
        save_features = input("Save decoded features to disk? (y/N): ").lower().strip() == 'y'
        if save_features:
            command.append("--save_features")
            
        self.run_command(command, "Semantic Understanding")
        
    def language_guided_editing(self):
        """Language-guided Scene Editing"""
        print("\n" + "="*60)
        print("✏️  Language-guided Scene Editing")
        print("="*60)
        
        # Find config files
        config_files = glob.glob(os.path.join(self.project_root, "configs", "**", "*.yaml"), recursive=True)
        config_files = [os.path.relpath(f, self.project_root) for f in config_files]
        
        if not config_files:
            print("❌ No config files found!")
            return
            
        # Automatically select option 2 (configs/wild/davis.yaml) as default
        default_config = "configs/wild/davis.yaml"
        if default_config in config_files:
            config_path = default_config
            print(f"✅ Using default config: {config_path}")
        else:
            # Fallback to first config if default not found
            config_path = config_files[0]
            print(f"⚠️  Default config not found, using: {config_path}")
                
        # Get root directory - show options under ./output
        output_dirs = glob.glob(os.path.join(self.project_root, "output", "*"))
        output_dirs = [d for d in output_dirs if os.path.isdir(d)]  # Only directories
        
        if not output_dirs:
            print("❌ No directories found in ./output!")
            return
            
        print("\n📁 Available output directories:")
        relative_dirs = []
        for i, dir_path in enumerate(output_dirs, 1):
            rel_path = os.path.relpath(dir_path, self.project_root)
            relative_dirs.append(rel_path)
            print(f"  {i}. {rel_path}")
            
        # Select root directory
        while True:
            try:
                choice = input(f"\nSelect root directory (1-{len(relative_dirs)}): ").strip()
                choice = int(choice)
                if 1 <= choice <= len(relative_dirs):
                    root_dir = relative_dirs[choice - 1]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(relative_dirs)}")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Get user prompt
        print("\nExample prompts:")
        print("  - 'Delete the car'")
        print("  - 'Extract the car'") 
        print("  - 'Make the car blue'")
        print("  - 'Make the color of the car look like Bumblebee'")
        
        user_prompt = input("\nEnter your editing prompt: ").strip()
        
        # Get advanced parameters with default suggestions
        print("\n⚙️  Advanced Hyperparameter Tuning Settings:")
        print("The editing agent will automatically try different threshold values to achieve the best results.")
        print("You can control the search space and number of attempts:")
        print("• Lower bound: Minimum threshold value to try (more inclusive, may include unwanted areas)")
        print("• Upper bound: Maximum threshold value to try (more exclusive, may miss target areas)")
        print("• Number of attempts: How many different threshold values to test between bounds")
        print("Press Enter to use recommended defaults, or specify custom values:")
        
        # Threshold lower bound
        threshold_lb_input = input(f"\nThreshold lower bound (0.0-1.0) [default: 0.85]: ").strip()
        threshold_lb = float(threshold_lb_input) if threshold_lb_input else 0.85
        
        # Threshold upper bound
        threshold_ub_input = input(f"Threshold upper bound (0.0-1.0) [default: 0.95]: ").strip()
        threshold_ub = float(threshold_ub_input) if threshold_ub_input else 0.95

        # Number of attempts
        num_attempt_input = input(f"Number of tuning attempts between thresholds [default: 10]: ").strip()
        num_attempt = int(num_attempt_input) if num_attempt_input else 10
        
        print(f"\n📊 Tuning Configuration:")
        print(f"   • Will test {num_attempt} different threshold values")
        print(f"   • Threshold range: {threshold_lb:.2f} to {threshold_ub:.2f}")
        print(f"   • The agent will automatically find the best threshold for your edit")
        
        # Get OpenAI API key
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            print("\n🔐 Using OPENAI_API_KEY from environment variables")
        else:
            print("\n🔐 OpenAI API Key Required")
            print("You need an OpenAI API key for language-guided editing.")
            print("")
            print("💡 Setup Options:")
            print("   1. Set environment variable: export OPENAI_API_KEY=your_api_key_here")
            print("   2. Add to your ~/.bashrc: echo 'export OPENAI_API_KEY=your_api_key_here' >> ~/.bashrc")
            print("   3. Or enter it now (one-time use)")
            print("")
            print("Get your API key at: https://platform.openai.com/account/api-keys")
            api_key = input("\nEnter your OpenAI API key: ").strip()
            
            if not api_key:
                print("❌ OpenAI API key is required for language-guided editing!")
                print("💡 Tip: Set OPENAI_API_KEY environment variable to avoid entering it each time.")
                return
        
        command = [
            "python", "agent_editing.py",
            "--config", config_path,
            "--root", root_dir,
            "--user_prompt", user_prompt,
            "--num_attempt", str(num_attempt),
            "--threshold_lb", str(threshold_lb),
            "--threshold_ub", str(threshold_ub)
        ]
        
        # Only add API key argument if not using environment variable
        if not os.getenv('OPENAI_API_KEY'):
            command.extend(["--api_key", api_key])
        
        self.run_command(command, "Language-guided Editing")
        
    def visual_question_answering(self):
        """4D Visual Question Answering"""
        print("\n" + "="*60)
        print("💬 4D Visual Question Answering (VQA)")
        print("="*60)
        
        # Extract data names from available rendered results
        rendered_files = self.find_rendered_results()
        if not rendered_files:
            print("❌ No rendered results found!")
            return
            
        # Extract unique data names from paths like "output/bear/final_viz/*/rendered_results.pth"
        data_names = set()
        for file in rendered_files:
            if file.startswith("output/"):
                parts = file.split("/")
                if len(parts) >= 2:
                    data_names.add(parts[1])  # Extract "bear" from "output/bear/..."
                    
        if not data_names:
            print("❌ No valid data found in output directory!")
            return
            
        data_names = sorted(list(data_names))
        
        print("📂 Available datasets:")
        for i, name in enumerate(data_names, 1):
            print(f"  {i}. {name}")
            
        # Select dataset
        while True:
            try:
                choice = input(f"\nSelect dataset (1-{len(data_names)}): ").strip()
                choice = int(choice)
                if 1 <= choice <= len(data_names):
                    data_name = data_names[choice - 1]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(data_names)}")
            except ValueError:
                print("❌ Please enter a valid number")
                
        # Find available views for the selected dataset
        base_dir = os.path.join(self.project_root, "output", data_name, "final_viz")
        views = []
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                view_path = os.path.join(base_dir, item)
                if os.path.isdir(view_path) and os.path.exists(os.path.join(view_path, "rendered_results.pth")):
                    views.append(item)
                    
        if not views:
            print(f"❌ No rendered results found for {data_name}!")
            return
            
        views = sorted(views)
        print(f"\n🎭 Available views for {data_name}:")
        for i, view in enumerate(views, 1):
            print(f"  {i}. {view}")
            
        # Select view
        while True:
            try:
                choice = input(f"\nSelect view (1-{len(views)}): ").strip()
                choice = int(choice)
                if 1 <= choice <= len(views):
                    view = views[choice - 1]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(views)}")
            except ValueError:
                print("❌ Please enter a valid number")
                
        print(f"\n✅ Using dataset: {data_name}")
        print(f"✅ Using view: {view}")
        
        # Check for Hugging Face token in environment first
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            print("\n🔐 Using HF_TOKEN from environment variables")
        else:
            print("\n🔐 Hugging Face Access Token")
            print("You need a Hugging Face access token to use InternVideo2-Chat-8B.")
            print("")
            print("💡 Setup Options:")
            print("   1. Set environment variable: export HF_TOKEN=your_token_here")
            print("   2. Add to your ~/.bashrc: echo 'export HF_TOKEN=your_token_here' >> ~/.bashrc")
            print("   3. Or enter it now (one-time use)")
            print("")
            print("Get your token at: https://huggingface.co/settings/tokens")
            hf_token = input("\nEnter your Hugging Face access token: ").strip()
            
            if not hf_token:
                print("❌ Access token is required for VQA functionality!")
                print("💡 Tip: Set HF_TOKEN environment variable to avoid entering it each time.")
                return
        
        command = [
            "python", "agent_chat.py",
            "--data_name", data_name,
            "--rendered_view", view,
            "--interactive"
        ]
        
        # Only add token argument if not using environment variable
        if not os.getenv('HF_TOKEN'):
            command.extend(["--huggingface_token", hf_token])
        
        self.run_command(command, "4D Visual Question Answering")
        
    def display_main_menu(self):
        """Display the main menu"""
        print("\n" + "="*60)
        print("🌟 Feature4X - Interactive Interface")
        print("="*60)
        print("Choose a functionality:")
        print()
        print("1. 🎯 SAM2 Segmentation")
        print("   Interactive segmentation using SAM2 with 4D feature fields")
        print()
        print("2. 🧠 Semantic Understanding")
        print("   Language-guided semantic segmentation with LSeg")
        print()
        print("3. ✏️  Language-guided Editing")
        print("   Advanced scene editing with natural language commands")
        print()
        print("4. 💬 4D Visual Question Answering (VQA)")
        print("   Interactive chat with 4D scenes")
        print()
        print("5. ❓ Help & Information")
        print()
        print("0. 🚪 Exit")
        print("="*60)
        
    def show_help(self):
        """Show help information"""
        print("\n" + "="*60)
        print("❓ Feature4X Help & Information")
        print("="*60)
        print()
        print("🔧 Prerequisites:")
        print("  - Conda environment 'feature4x' should be activated")
        print("  - Run this script from the Feature4X project root directory")
        print("  - Ensure you have processed data (rendered_results.pth files)")
        print()
        print("📁 File Structure:")
        print("  - Rendered results: output/**/rendered_results.pth")
        print("  - Config files: configs/**/*.yaml") 
        print("  - Feature files: data/**/internvideo_feats.pth")
        print()
        print("🎯 SAM2 Segmentation:")
        print("  - Interactive coordinate selection on reference image")
        print("  - Support for point and/or bounding box input")
        print("  - Generates segmentation masks across all frames")
        print()
        print("🧠 Semantic Understanding:")
        print("  - Language-guided semantic segmentation")
        print("  - Optional custom labels and feature saving")
        print("  - Generates videos and visualizations")
        print()
        print("✏️  Language-guided Editing:")
        print("  - Natural language scene editing commands")
        print("  - Requires OpenAI API key for LLM processing")
        print("  - Examples: 'Delete the bear', 'Change color to red'")
        print()
        print("💬 4D VQA:")
        print("  - Interactive chat with 4D scenes")
        print("  - Ask questions about scene content and dynamics")
        print("  - Requires both rendered results and feature files")
        print()
        input("Press Enter to continue...")
        
    def run(self):
        """Main interactive loop"""
        print("🌟 Welcome to Feature4X Interactive Interface!")
        print("Bridging Any Monocular Video to 4D Agentic AI with Versatile Gaussian Feature Fields")
        
        self.check_conda_env()
        
        while True:
            self.display_main_menu()
            
            choice = input("Enter your choice (0-5): ").strip()
            
            if choice == "0":
                print("\n👋 Thank you for using Feature4X!")
                break
            elif choice == "1":
                self.sam2_segmentation()
            elif choice == "2":
                self.semantic_understanding()
            elif choice == "3":
                self.language_guided_editing()
            elif choice == "4":
                self.visual_question_answering()
            elif choice == "5":
                self.show_help()
            else:
                print("\n❌ Invalid choice! Please enter a number from 0-5.")
                
            if choice != "0" and choice != "5":
                input("\nPress Enter to return to main menu...")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Feature4X Interactive Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python feature4x_interactive.py
  
Make sure to:
  1. Activate conda environment: conda activate feature4x
  2. Run from Feature4X project root directory
  3. Have processed data available (rendered_results.pth files)
        """
    )
    
    args = parser.parse_args()
    
    try:
        interface = Feature4XInterface()
        interface.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()