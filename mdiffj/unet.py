"""Simple UNet implementation.

https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
"""

import dataclasses
from typing import Callable, Literal, Optional, Tuple

import jax
from flax import linen as nn
from jax import numpy as jnp
from jaxtyping import Float

relu_init = jax.nn.initializers.kaiming_normal()


class UNet(nn.Module):
    scales: int = 5
    base_feature_dim: int = 64
    encoder_dropout_layers: Tuple[int, ...] = (3, 4)
    encoder_dropout_rate: float = 0.5
    output_channels: int = 1
    output_activation: Optional[Literal["sigmoid", "tanh"]] = None

    @nn.compact
    def __call__(
        self, x: Float[jnp.ndarray, "b h w c"], training: bool = False
    ) -> Float[jnp.ndarray, "b h w output_channels"]:
        # Encode.
        encoded: List[jnp.ndarray] = []
        for scale_idx in range(self.scales):
            feature_dim = self.base_feature_dim * (2**scale_idx)
            x = nn.Conv(
                feature_dim,
                kernel_size=(3, 3),
                kernel_init=relu_init,
                use_bias=True,
            )(x)
            x = nn.relu(x)
            x = nn.Conv(
                feature_dim,
                kernel_size=(3, 3),
                kernel_init=relu_init,
                use_bias=False,
            )(x)
            # x = nn.LayerNorm()(x)  # type: ignore
            x = nn.relu(x)

            if scale_idx in self.encoder_dropout_layers:
                x = nn.Dropout(
                    rate=self.encoder_dropout_rate, deterministic=not training
                )(x)
            if scale_idx != self.scales - 1:
                encoded.append(x)
                x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Decode.
        for scale_idx in reversed(range(self.scales - 1)):
            feature_dim = self.base_feature_dim * (2**scale_idx)

            # Upsample.
            x = jax.image.resize(
                x,
                shape=(
                    x.shape[0],
                    x.shape[1] * 2,
                    x.shape[2] * 2,
                    x.shape[3],
                ),
                method="linear",  # or nearest?
            )
            x = nn.Conv(
                feature_dim,
                kernel_size=(2, 2),
                kernel_init=relu_init,
                use_bias=True,
            )(x)
            x = nn.relu(x)
            x = jnp.concatenate([encoded[scale_idx], x], axis=3)  # type: ignore
            x = nn.Conv(
                feature_dim,
                kernel_size=(3, 3),
                kernel_init=relu_init,
                use_bias=True,
            )(x)
            x = nn.relu(x)
            x = nn.Conv(
                feature_dim,
                kernel_size=(3, 3),
                kernel_init=relu_init,
                use_bias=False,
            )(x)
            # x = nn.LayerNorm()(x)  # type: ignore
            x = nn.relu(x)

        x = nn.Conv(self.output_channels, kernel_size=(1, 1))(x)
        if self.output_activation is not None:
            x = {"tanh": nn.tanh, "sigmoid": nn.sigmoid}[self.output_activation](x)
        return x


if __name__ == "__main__":
    import tyro

    print("making model")
    model = tyro.cli(UNet)
    print("making input")
    x = jnp.zeros((10, 32, 32, 3))
    print("input shape:", x.shape)
    print("initializing")
    params = jax.jit(model.init)(jax.random.PRNGKey(0), x)
    print("forward pass")
    print("output shape:", model.apply(params, x).shape)
