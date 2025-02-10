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
```

### Usage
You need sufficiently powerful hardware to train the model.
1. run get_images.py
You can specify the parameters either through the get_images.py file directly or through argparse:
```bash
python get_images.py --dataset STL10 --save_folder ./data/images --num_images 1000 --img_size 256
```
```bash
python get_images.py --help
```
2. run main.py
You can specify the parameters either through the config.py file or through argparse:
```bash
python main.py --batch_size 16 --img_size 128 --epochs 20 --learning_rate 0.0005 --seed 1234 --checkpoint_path checkpoints/new_model.pkl
```
```bash
python main.py --help
```

