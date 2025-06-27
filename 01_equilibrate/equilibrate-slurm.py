import pickle
import click

from openff.units import unit
from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.evaluator.properties import Density, EnthalpyOfMixing
from openff.evaluator.client import RequestOptions

from openff.evaluator.backends import ComputeResources, QueueWorkerResources
from openff.evaluator.backends.dask import DaskLocalCluster
from openff.evaluator.backends.dask import DaskSLURMBackend

from openff.evaluator.client import EvaluatorClient, RequestOptions, ConnectionOptions
from openff.evaluator.server.server import EvaluatorServer
from openff.evaluator.layers.equilibration import EquilibrationProperty
from openff.evaluator.utils.observables import ObservableType

from openff.evaluator.forcefield import SmirnoffForceFieldSource


@click.command()
@click.option(
    "--dataset",
    "-d",
    "dataset_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default="dataset.json",
)
@click.option(
    "--n-molecules",
    "-n",
    type=int,
    default=1000,
)
@click.option(
    "--force-field",
    "-f",
    default="openff-2.2.1.offxml",
)
@click.option(
    "--port",
    "-p",
    default=8000,
)
@click.option(
    "--extra-script-option",
    "extra_script_options",
    type=str,
    multiple=True,
    help="Extra options to pass to the script",
)
@click.option(
    "--conda-env",
    default="sage-2.3.0-vdw",
    help="Conda environment to activate",
)
@click.option(
    "--n-gpu",
    default=60,
    help="Number of GPUs to use",
)
@click.option(
    "--queue",
    default="gpu",
    help="Queue to use",
)
def main(
    dataset_path: str,
    n_molecules: int = 1000,
    force_field: str = "openff-2.2.1.offxml",
    port: int = 8000,
    extra_script_options: list[str] = [],
    conda_env: str = "sage-2.3.0-vdw",
    n_gpu: int = 60,
    queue: str = "gpu",
):
    # load dataset
    dataset = PhysicalPropertyDataSet.from_json(dataset_path)
    print(f"Loaded {len(dataset.properties)} properties from {dataset_path}")

    potential_energy = EquilibrationProperty()
    potential_energy.relative_tolerance = 0.05
    potential_energy.observable_type = ObservableType.PotentialEnergy
    potential_energy.n_uncorrelated_samples = 300

    density = EquilibrationProperty()
    density.relative_tolerance = 0.05
    density.observable_type = ObservableType.Density
    density.n_uncorrelated_samples = 300

    options = RequestOptions()
    options.calculation_layers = ["EquilibrationLayer"]
    density_schema = Density.default_equilibration_schema(
        n_molecules=n_molecules,
        error_tolerances=[potential_energy, density],
        # every iteration is 200 ps
        max_iterations=60, # go up to 12 ns
        error_on_failure=False,
    )

    dhmix_schema = EnthalpyOfMixing.default_equilibration_schema(
        n_molecules=n_molecules,
        error_tolerances=[potential_energy, density],
        max_iterations=60,
        error_on_failure=False,
    )

    # note: output frequency is every 10 ps.

    options.add_schema("EquilibrationLayer", "Density", density_schema)
    options.add_schema("EquilibrationLayer", "EnthalpyOfMixing", dhmix_schema)
    
    force_field_source = SmirnoffForceFieldSource.from_path(
        force_field
    )

    worker_resources = QueueWorkerResources(
        number_of_threads=1,
        number_of_gpus=1,
        preferred_gpu_toolkit=ComputeResources.GPUToolkit.CUDA,
        per_thread_memory_limit=4 * unit.gigabyte,
        wallclock_time_limit="48:00:00",
    )

    backend = DaskSLURMBackend(
        minimum_number_of_workers=1,
        maximum_number_of_workers=n_gpu,  # 24 max on free queue -- keep 1 free.
        resources_per_worker=worker_resources,
        queue_name=queue,
        setup_script_commands=[
            "source ~/.bashrc",
            f"conda activate {conda_env}",
            "conda env export > conda-env.yaml",
        ],
        extra_script_options=extra_script_options,
        adaptive_interval="1000ms",
    )
    backend.start()
    print("backend", backend)

    server = EvaluatorServer(
        calculation_backend=backend,
        working_directory="working-directory",
        delete_working_files=False,
        port=port,
    )
    server.start(asynchronous=True)
    client = EvaluatorClient(
        connection_options=ConnectionOptions(server_port=port)
    )

    # we first request the equilibration data
    # this can be copied between different runs to avoid re-running
    # the data is saved in a directory called "stored_data"

    request, error = client.request_estimate(
        dataset,
        force_field_source,
        options,
    )
    print(request.id)

    # block until computation finished
    results, exception = request.results(synchronous=True, polling_interval=30)
    assert exception is None

    print(f"Equilibration complete")
    print(f"# estimated: {len(results.estimated_properties)}")
    print(f"# equilibrated: {len(results.equilibrated_properties)}")
    print(f"# unsuccessful: {len(results.unsuccessful_properties)}")
    print(f"# exceptions: {len(results.exceptions)}")

    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()

