from flax import linen as nn
import jax.numpy as jnp

class DownBlock(nn.Module):
    """Downsampling block with two convolutions followed by max pooling."""
    features: int

    @nn.compact
    def __call__(self, x):
        # First convolution + activation
        x = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        # Second convolution + activation
        x = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        skip = x  # Save output for skip connection

        # Downsample with max pooling
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')
        return skip, x

class UpBlock(nn.Module):
    """Upsampling block with transpose convolution, concatenation with skip, then two convolutions."""
    features: int

    @nn.compact
    def __call__(self, x, skip):
        # Upsample using transpose convolution
        x = nn.ConvTranspose(self.features, kernel_size=(2, 2), strides=(2, 2), padding='SAME')(x)
        # Concatenate skip connection from encoder
        x = jnp.concatenate([x, skip], axis=-1)
        # First convolution + activation
        x = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        # Second convolution + activation
        x = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        return x

class UNet(nn.Module):
    """
    Full UNet for image colorization in Lab color space.
    
    Input: Grayscale image (L channel) with shape (batch, H, W, 1)
    Output: ab channels with shape (batch, H, W, 2)
    """
    @nn.compact
    def __call__(self, x):
        # Encoder path
        skip1, x = DownBlock(features=32)(x)       # (B, H/2,  W/2,  32)
        skip2, x = DownBlock(features=64)(x)       # (B, H/4,  W/4,  64)
        skip3, x = DownBlock(features=128)(x)      # (B, H/8,  W/8,  128)
        skip4, x = DownBlock(features=256)(x)      # (B, H/16, W/16, 256)

        # Bottleneck
        x = nn.Conv(512, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(512, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)

        # Decoder path
        x = UpBlock(features=256)(x, skip4)        # (B, H/8,  W/8,  256)
        x = UpBlock(features=128)(x, skip3)        # (B, H/4,  W/4,  128)
        x = UpBlock(features=64)(x, skip2)         # (B, H/2,  W/2,  64)
        x = UpBlock(features=32)(x, skip1)         # (B, H,    W,    32)

        # Final 1x1 convolution to produce ab channels
        x = nn.Conv(2, kernel_size=(1, 1), padding='SAME')(x)
        # Map outputs to [-1, 1]
        return jnp.tanh(x)

def create_model():
    return UNet()
