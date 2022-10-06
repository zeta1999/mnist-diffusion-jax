from __future__ import annotations

import dataclasses
import io
from math import isqrt
from typing import Any, Tuple, Union, cast

import fifteen
import flax
import jax
import jax_dataclasses as jdc
import numpy as onp
import optax
import PIL.Image
from flax import linen as nn
from jax import numpy as jnp
from jaxtyping import Float, Int
from matplotlib import pyplot as plt
from typing_extensions import Literal, assert_never

from .unet import UNet

VarianceSchedule = Literal["ho2020", "nichol2021"]


@jdc.pytree_dataclass
class DDPMSchedule:
    """Helper class for computing useful constants.

    Loose notation reference:
        https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

    Note that our indices are t=0...T-1 instead of t=1...T.
    """

    T: jdc.Static[int]
    r"""Number of steps."""

    beta_t: Float[jnp.ndarray, "T"]
    r"""Variance schedule."""

    alpha_t: Float[jnp.ndarray, "T"]
    r"""$1 - \beta_t$"""

    alpha_bar_t: Float[jnp.ndarray, "T"]
    r"""$\Prod_{j=1}^t (1 - \beta_j)$"""

    sigma_t: Float[jnp.ndarray, "T"]
    r"""Reverse process variance."""

    @staticmethod
    def setup(variance_schedule: VarianceSchedule) -> DDPMSchedule:
        if variance_schedule == "ho2020":
            T = 1000
            # From Sec 4 of Denoising Diffusion Probabilistic Models
            # https://arxiv.org/abs/2006.11239
            beta_t = jnp.linspace(start=1e-4, stop=0.02, num=T)
        elif variance_schedule == "nichol2021":
            # TODO
            assert False
        else:
            assert_never(variance_schedule)

        assert beta_t.shape == (T,)
        sqrt_beta_t = jnp.sqrt(beta_t)
        alpha_t = 1.0 - beta_t
        alpha_bar_t = jnp.exp(jnp.cumsum(jnp.log(alpha_t)))
        sigma_t = jnp.sqrt(beta_t)

        return DDPMSchedule(
            T,
            beta_t,
            alpha_t,
            alpha_bar_t,
            sigma_t,
        )


@dataclasses.dataclass(frozen=True)
class DDPMConfig:
    variance_schedule: VarianceSchedule = "ho2020"

    eps_network: UNet = dataclasses.field(
        default_factory=lambda: UNet(
            scales=4,
            encoder_dropout_layers=(2, 3),
            encoder_dropout_rate=0.3,
            output_channels=1,
            output_activation=None,
        )
    )
    """Epsilon network."""

    learning_rate: float = 1e-4
    """ADAM learning rate."""

    seed: int = 94709


FlaxParams = flax.core.FrozenDict[str, Any]


@jdc.pytree_dataclass
class DDPMState:
    constants: DDPMSchedule

    eps_network: jdc.Static[nn.Module]
    eps_params: FlaxParams

    opt_tx: jdc.Static[optax.GradientTransformation]
    opt_state: optax.OptState

    prng: jax.random.KeyArray
    steps: Int[jnp.ndarray, ""]  # Scalar integer.

    @staticmethod
    def setup(config: DDPMConfig) -> DDPMState:
        prng_params, prng_state = jax.random.split(jax.random.PRNGKey(config.seed))

        dummy_x = jnp.zeros((1, 32, 32, 2))  # Last channel is intensity, step #
        eps_params = config.eps_network.init(prng_params, dummy_x)

        opt_tx = optax.adam(learning_rate=config.learning_rate)
        opt_state = opt_tx.init(eps_params)
        return DDPMState(
            constants=DDPMSchedule.setup(config.variance_schedule),
            eps_network=config.eps_network,
            eps_params=eps_params,
            opt_tx=opt_tx,
            opt_state=opt_state,
            prng=prng_state,
            steps=jnp.array(0, dtype=jnp.int32),  # type: ignore
        )

    # Wrapper for typing; we lose annotations + tab complete under @jax.jit.
    # This is kind of gross.
    def train_step(
        self, x: Float[jnp.ndarray, "b h w c"]
    ) -> Tuple[DDPMState, fifteen.experiments.TensorboardLogData]:
        return self._train_step(x)

    @jax.jit
    def _train_step(
        self, x: Float[jnp.ndarray, "b h w c"]
    ) -> Tuple[DDPMState, fifteen.experiments.TensorboardLogData]:
        prng_ts, prng_eps, prng_new = jax.random.split(self.prng, 3)
        b, h, w, c = x.shape

        # Sample via the "nice property".
        ts = jax.random.randint(
            key=prng_ts,
            shape=(b,),
            minval=0,
            maxval=self.constants.T,
        )
        eps = jax.random.normal(key=prng_eps, shape=x.shape, dtype=x.dtype)
        x_t = (
            jnp.sqrt(self.constants.alpha_bar_t[ts, None, None, None]) * x
            + jnp.sqrt(1.0 - self.constants.alpha_bar_t[ts, None, None, None]) * eps
        )

        # Define loss.
        def train_loss(eps_params: FlaxParams):
            eps_pred = self.eps_network.apply(
                eps_params,
                self.step_condition(x_t, ts),
            )
            assert isinstance(eps_pred, jnp.ndarray)

            # Data to log to Tensorboard.
            loss = jnp.mean((eps_pred - eps) ** 2)
            log_data = fifteen.experiments.TensorboardLogData(
                scalars={"loss": loss},
                histograms={"eps_pred_0": eps_pred[0]},
            )

            return loss, log_data

        # Compute loss & gradients with respect to parameters.
        (loss, log_data), eps_grads = jax.value_and_grad(train_loss, has_aux=True)(
            self.eps_params
        )

        # ADAM update.
        updates, new_opt_state = self.opt_tx.update(eps_grads, self.opt_state)

        with jdc.copy_and_mutate(self) as updated:
            updated.eps_params = cast(
                FlaxParams, optax.apply_updates(self.eps_params, updates)
            )
            updated.opt_state = new_opt_state
            updated.prng = prng_new
            updated.steps = updated.steps + 1

        return updated, log_data

    @jax.jit
    def denoise_step(
        self,
        x_t: Float[jnp.ndarray, "b h w c"],
        ts: Int[jnp.ndarray, "b"],
        prng: jax.random.KeyArray,
    ) -> Float[jnp.ndarray, "b h w c"]:
        """Compute x_{t+1} from x_{t}."""
        b, h, w, c = x_t.shape
        assert ts.shape == (b,)

        z = jax.random.normal(prng, shape=x_t.shape)
        eps_t = cast(
            jnp.ndarray,
            self.eps_network.apply(
                self.eps_params,
                self.step_condition(x_t, ts),
            ),
        )
        assert eps_t.shape == (b, h, w, c)

        sigma_t = self.constants.sigma_t[ts, None, None, None]
        weighted_eps = (
            (1.0 - self.constants.alpha_t[ts])
            / jnp.sqrt(1.0 - self.constants.alpha_bar_t[ts])
        )[:, None, None, None] * eps_t

        x_t_plus_1 = (x_t - weighted_eps) / jnp.sqrt(
            self.constants.alpha_t[ts, None, None, None]
        ) + sigma_t * z
        assert x_t_plus_1.shape == x_t.shape

        return x_t_plus_1

    def step_condition(
        self, x_t: Float[jnp.ndarray, "b h w c"], ts: Int[jnp.ndarray, "b"]
    ) -> Float[jnp.ndarray, "b h w c+1"]:
        b, h, w, c = x_t.shape
        return jnp.concatenate(
            [
                x_t,
                jnp.tile(ts[:, None, None, None], reps=(1, h, w, 1))
                / (self.constants.T - 1),
            ],
            axis=-1,
        )  # type: ignore

    def visualize_samples(
        self,
        num_samples: Literal[4, 9, 16, 25, 36, 49, 64],
        seed: int,
    ) -> onp.ndarray:
        # Run denoising steps, from x_T to x_0.
        prng_seq = jax.random.split(
            jax.random.PRNGKey(seed),
            self.constants.T + 1,
        )
        x = jax.random.normal(prng_seq[0], shape=(num_samples, 32, 32, 1))
        print("Rendering...", end="", flush=True)
        for t in tuple(reversed(range(self.constants.T))):
            ts = t * onp.ones(num_samples, dtype=jnp.int32)
            x = self.denoise_step(x, ts, prng=prng_seq[t + 1])
        print(" done!")

        # Plot.
        dim = isqrt(num_samples)
        fig, axs = plt.subplots(dim, dim, figsize=(8, 8))
        axs = axs.flatten()
        for i in range(num_samples):
            axs[i].axis("off")
            axs[i].imshow(x[i])

        buf = io.BytesIO()
        plt.savefig(buf, format="jpeg")
        buf.seek(0)
        return onp.array(PIL.Image.open(buf))
