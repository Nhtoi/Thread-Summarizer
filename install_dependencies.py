import subprocess
import sys

def install_dependencies():
    """Install all required dependencies for the Reddit Thread Summarizer."""
    
    print("Installing required dependencies...")
    
    # List of dependencies
    dependencies = [
        "flask",
        "praw",
        "python-dotenv",
        "pandas",
        "numpy",
        "torch",
        "transformers",
        "datasets",
        "tqdm",
        "matplotlib",
        "scikit-learn",
        "sentencepiece"  # This is critical for T5 tokenizer
    ]
    
    # Install each dependency
    for dep in dependencies:
        print(f"\nInstalling {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"Successfully installed {dep}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {dep}")
    
    print("\nDependency installation complete. You can now run app.py")

if __name__ == "__main__":
    install_dependencies()