import dataclasses
import pathlib

import fifteen
import jax
import tyro
from rich import print
from rich.console import Group
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from rich.text import Text

from mdiffj import ddpm_state, mnist_data


@dataclasses.dataclass
class TrainLoopConfig:
    minibatch_size: int = 32
    num_iterations: int = 100_000


def main(
    experiment_name: str,
    timestamp_experiment: bool = False,
    output_dir: pathlib.Path = pathlib.Path("./outputs"),
    train_config: TrainLoopConfig = TrainLoopConfig(),
    ddpm_config: ddpm_state.DDPMConfig = ddpm_state.DDPMConfig(),
) -> None:
    # Setup.
    state = ddpm_state.DDPMState.setup(ddpm_config)
    dataset = mnist_data.load_mnist_dataset("train")
    dataloader = fifteen.data.InMemoryDataLoader(
        dataset,
        minibatch_size=train_config.minibatch_size,
        drop_last=True,
    )

    # Experiment setup.
    if timestamp_experiment:
        experiment_name = f"{fifteen.utils.timestamp()}-{experiment_name}"
    exp = fifteen.experiments.Experiment(data_dir=output_dir / experiment_name)
    exp.write_metadata("train_config", train_config)
    exp.write_metadata("ddpm_config", ddpm_config)

    print("Training with config:")
    print(train_config)
    print(ddpm_config)

    # Training loop.
    status = Status("Starting...", refresh_per_second=10.0)
    status.start()
    for metrics, minibatch in zip(
        fifteen.utils.loop_metric_generator(),
        fifteen.data.cycled_minibatches(dataloader, shuffle_seed=0),
    ):
        # Step.
        state, log_data = state.train_step(minibatch.image)

        # Update CLI status.
        status.update(
            Panel(
                Group(
                    Pretty(metrics),
                    Pretty(log_data.scalars),
                )
            )
        )

        # Logging and checkpointing.
        step = int(state.steps)
        exp.log(
            log_data,
            step=step,
            log_scalars_every_n=30,
            log_histograms_every_n=100,
        )
        if step % 1000 == 0:
            exp.save_checkpoint(state, step=step)
        if step % 2000 == 0:
            exp.summary_writer.image(
                "samples",
                state.visualize_samples(num_samples=25, seed=94709),
                step=state.steps,
            )

        # Break if done!
        if step >= train_config.num_iterations:
            break
    status.stop()


if __name__ == "__main__":
    fifteen.utils.pdb_safety_net()
    tyro.cli(main)
