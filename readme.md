# JAX Image Colorization

A UNet implementation for automatic image colorization using JAX and Flax.

## Setup Instructions

### Windows
- Run `setup.bat` (double-click it or execute via Command Prompt). This will create a virtual environment and install all dependencies.

### Linux / macOS
Open a terminal in the project directory and run:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

## Usage
You need sufficiently powerful hardware to train the model.
1. run get_images.py
2. run main.py