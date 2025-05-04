import os
import jax
from jax.lib import xla_bridge

# 🚀 Force JAX to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3"

# 🔍 Check JAX backend and devices
print("CUDNN_INCLUDE_DIR:", os.environ.get("CUDNN_INCLUDE_DIR"))
print("CUDNN_LIB_DIR:", os.environ.get("CUDNN_LIB_DIR"))
print("JAX backend:", xla_bridge.get_backend().platform)
print("JAX devices:", jax.devices())

# 🛠 Perform a small GPU computation test
import jax.numpy as jnp

x = jnp.ones((3, 3))
print("Computation device:", jax.devices()[0])
print("Matrix multiplication test:", jnp.dot(x, x))
