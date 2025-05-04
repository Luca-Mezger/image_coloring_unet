@echo off
REM This script automates the setup and debugging of JAX with GPU support.

REM Step 1: Automatically find the CUDA installation path
echo Searching for CUDA installation...
set CUDA_HOME=
for /d %%i in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do set CUDA_HOME=%%i

if not defined CUDA_HOME (
    echo CUDA is not installed. Please install CUDA and NVIDIA drivers.
    pause
    exit /b 1
)

REM Step 2: Automatically find cuDNN installation path
echo Searching for cuDNN installation...
set CUDNN_INCLUDE_DIR=
set CUDNN_LIB_DIR=
for /d %%i in ("C:\tools\cuda") do (
    set CUDNN_INCLUDE_DIR=%%i\include
    set CUDNN_LIB_DIR=%%i\lib
)

if not defined CUDNN_INCLUDE_DIR (
    echo cuDNN is not installed. Please install cuDNN.
    pause
    exit /b 1
)

REM Step 3: Set environment variables for CUDA and cuDNN
set PATH=%CUDA_HOME%\bin;%PATH%
set LIBRARY_PATH=%CUDA_HOME%\libnvvp;%LIBRARY_PATH%
set CUDNN_INCLUDE_DIR=%CUDNN_INCLUDE_DIR%
set CUDNN_LIB_DIR=%CUDNN_LIB_DIR%

REM Step 4: Check if Python is installed
python --version > nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not found in PATH. Please install Python 3.10 or higher and ensure it's added to PATH.
    pause
    exit /b 1
)

REM Step 5: Check if NVIDIA drivers and CUDA toolkit are installed
echo Checking if CUDA is installed...
nvidia-smi > nul 2>&1
if errorlevel 1 (
    echo CUDA is not installed or NVIDIA drivers are not detected. Please ensure CUDA and NVIDIA drivers are installed.
    pause
    exit /b 1
)

REM Step 6: Check CUDA version
echo Checking CUDA version...
nvcc --version

REM Step 7: Check if JAX is installed
python -c "import jax" > nul 2>&1
if errorlevel 1 (
    echo JAX is not installed. Installing JAX...
    pip install --upgrade pip
    pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
)

REM Step 8: Check if correct JAX version is installed
echo Checking JAX version and device...
python -c "import jax; print('JAX version:', jax.__version__); from jax.lib import xla_bridge; print('JAX backend:', xla_bridge.get_backend().platform); print('JAX devices:', jax.devices())"

REM Step 9: Check if JAX is using the GPU
echo Verifying if JAX detects the GPU...
python -c "import jax; from jax.lib import xla_bridge; print('JAX GPU status:', xla_bridge.get_backend().platform); print('JAX devices:', jax.devices())"

REM Step 10: Debugging - If JAX is still using CPU, run this diagnostic:
echo If JAX is still using CPU, the following steps might help:
echo 1. Check that CUDA and cuDNN versions are compatible with JAX and your Python version.
echo 2. Ensure the environment variables for CUDA and cuDNN are set.
echo 3. Check if Python 3.10 is correctly configured in the environment.
echo 4. If you have more than one Python version, make sure that the correct version is being used (Python 3.10 or higher).
echo 5. Consider reinstalling the JAX dependencies for GPU support.
pause

REM Step 11: Create a virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Step 12: Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Step 13: Upgrade pip inside the virtual environment
echo Upgrading pip inside virtual environment...
pip install --upgrade pip

REM Step 14: Install dependencies from requirements.txt
echo Installing dependencies from requirements.txt...
if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo No requirements.txt file found. Skipping this step.
)

REM Step 15: Check for any Python environment-related issues
echo Verifying Python environment...
python -c "import sys; print('Python Executable:', sys.executable); print('Python Path:', sys.path)"

echo Setup complete. To activate the environment later, run: venv\Scripts\activate.bat
pause
