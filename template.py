# Import necessary modules
import os
from pathlib import Path
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Define the project name
project_name = "cnnClassifier"

# List of files to be created or checked
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html"
]

# Iterate through the list of files
for filepath in list_of_files:
    # Convert the file path to a Path object
    filepath = Path(filepath)
    
    # Extract the directory and filename from the path
    filedir, filename = os.path.split(filepath)

    # If the directory is not empty, create it (and any necessary parent directories)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        # Log a message indicating the creation of the directory
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    # If the file does not exist or has size 0, create an empty file
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        # Log a message indicating the creation of an empty file
        logging.info(f"Creating empty file: {filepath}")

    # If the file already exists, log a message
    else:
        logging.info(f"{filename} already exists")
