import os
import pickle
import click
import pathlib

from openff.units import unit
from openff.evaluator.backends import ComputeResources, QueueWorkerResources
from openff.evaluator.backends.dask import DaskLocalCluster, DaskSLURMBackend
from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.evaluator.client import RequestOptions
from openff.evaluator.backends import ComputeResources
from openff.evaluator.storage.localfile import LocalFileStorage

from openff.evaluator.client import EvaluatorClient, RequestOptions, ConnectionOptions
from openff.evaluator.server.server import EvaluatorServer

from openff.evaluator.forcefield import SmirnoffForceFieldSource
from openff.toolkit import ForceField


@click.command()
@click.option(
    "--input-path",
    "-i",
    default="../../../01_curate-physprop/validation/final/output/validation-set.json",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="Path to the input dataset.",
)
@click.option(
    "--storage-directory",
    "-s",
    type=click.Path(exists=True, dir_okay=True),
    help="Storage directory",
)
@click.option(
    "--forcefield",
    "-ff",
    type=str,
    default="../../forcefields/openff-2.2.1.offxml",
    help="Path to the forcefield file.",
)
@click.option(
    "--output-directory",
    "-o",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="validation",
    help="Path to the output directory.",
)
@click.option(
    "--replicate",
    "-r",
    type=int,
    default=1,
    help="Replicate ID to run.",
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=8100,
    help="Base port to use for the server.",
)
@click.option(
    "--options-file",
    "-of",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    default="request-options.json",
    help="Path to the request options file.",
)
@click.option(
    "--conda-env",
    default="evaluator-050",
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
@click.option(
    "--extra-script-option",
    "extra_script_options",
    type=str,
    multiple=True,
    help="Extra options to pass to the script",
)
def main(
    input_path: str,
    forcefield: str,
    output_directory: str,
    storage_directory: str,
    replicate: int = 1,
    port: int = 8100,
    options_file: str = "request-options.json",
    conda_env: str = "evaluator-050",
    n_gpu: int = 60,
    queue: str = "gpu",
    extra_script_options: list[str] = [],
):
    small_dataset = PhysicalPropertyDataSet.from_json(input_path)
    print(len(small_dataset.properties))
    
    ff = ForceField(forcefield)
    ff_name = pathlib.Path(forcefield).stem
    force_field_source = SmirnoffForceFieldSource.from_object(ff)

    options = RequestOptions.from_json(options_file)

    output_directory = pathlib.Path(output_directory)
    storage_directory = LocalFileStorage(
        root_directory=str((output_directory / "stored_data").resolve()),
        cache_objects_in_memory=True,
    )
    
    output_directory = output_directory / f"rep-{replicate}" / ff_name
    output_directory.mkdir(parents=True, exist_ok=True)
    os.chdir(output_directory)

    pickle_file = f"properties.pkl"
    results_file = f"properties.json"

    if pathlib.Path(results_file).exists():
        print(f"{results_file} already exists")
        return
 
    print(f"Starting server on port {port}")

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

    if True:
        server = EvaluatorServer(
            calculation_backend=backend,
            working_directory="working-directory",
            delete_working_files=True,
            enable_data_caching=False,
            storage_backend=storage_directory,
            port=port,
        )
        with server:
            client = EvaluatorClient(
                connection_options=ConnectionOptions(server_port=port)
            )
            request, error = client.request_estimate(
                small_dataset,
                force_field_source,
                options
            )

            assert error is None
            results, exception = request.results(synchronous=True, polling_interval=30)

    assert exception is None

    print(f"Simulation complete")
    print(f"# estimated: {len(results.estimated_properties)}")
    print(f"# unsuccessful: {len(results.unsuccessful_properties)}")
    print(f"# exceptions: {len(results.exceptions)}")
    #pickle_file.parent.mkdir(exist_ok=True, parents=True)

    with open(pickle_file, "wb") as f:
        pickle.dump(results, f)
    print(f"Results dumped to {pickle_file}")

    results.json(results_file, format=True)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
