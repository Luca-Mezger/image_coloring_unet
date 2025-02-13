import jax
import jax.numpy as jnp

def check_jax_gpu():
    try:
        # Check if GPU is available
        devices = jax.devices()
        gpu_available = any("cuda" in device.device_kind.lower() for device in devices)
        
        if gpu_available:
            print("✅ JAX is using GPU!")
        else:
            print("❌ JAX is not using GPU. Running on CPU instead.")

        print("Devices:", devices)

        # Run a simple computation
        print("Testing JAX computation...")
        x = jnp.ones((1000, 1000))
        y = jnp.dot(x, x)
        print("Computation successful!", y.shape)

    except Exception as e:
        print("⚠️ Error while testing JAX with GPU:", e)

if __name__ == "__main__":
    check_jax_gpu()
