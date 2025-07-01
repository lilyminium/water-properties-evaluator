import pathlib

import click
import tqdm

import pandas as pd

from openff.evaluator.datasets.datasets import PhysicalPropertyDataSet
from openff.evaluator.server.server import RequestResult


@click.command()
@click.option(
    "--input-dataset",
    "-i",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    default="dataset.json",
    help="Path to the input dataset file.",
)
@click.option(
    "--input-directory",
    "-d",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="training",
    help="Directory containing the input JSON files.",
)
@click.option(
    "--output-directory",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    default="output",
    help="Directory to write the output CSV file to.",
)
def main(
    input_dataset: str = "dataset.json",
    input_directory: str = "training",
    output_directory: str = "output",
):
    
    all_json_files = pathlib.Path(input_directory).glob("rep*/*/*.json")

    reference_dataset = PhysicalPropertyDataSet.from_json(input_dataset)
    reference_properties_by_id = {
        prop.id: prop
        for prop in reference_dataset.properties
    }

    all_entries = []

    for json_file in tqdm.tqdm(sorted(all_json_files)):
        request_result = RequestResult.from_json(json_file)
        replicate = int(json_file.parent.parent.stem.split("-")[-1])
        for i, physical_property in enumerate(request_result.estimated_properties.properties): 
            value = physical_property.value.m
            uncertainty = physical_property.uncertainty.m
            reference_property = reference_properties_by_id[physical_property.id]
            ref_value = reference_property.value.m
            ref_uncertainty = reference_property.uncertainty.m
            try:
                index = int(json_file.stem.split("-")[-1]) + i
            except:
                index = i
            ff_name = json_file.parent.name
        
            entry = {
                "id": physical_property.id,
                "index": int(index),
                "type": type(physical_property).__name__,
                "forcefield": ff_name,
                "replicate": replicate,
                "value": value,
                "uncertainty": uncertainty,
                "reference_value": ref_value,
                "reference_uncertainty": ref_uncertainty,
            }
            all_entries.append(entry)

    df = pd.DataFrame(all_entries)
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_file = output_directory / "benchmarks.csv"
    df.to_csv(csv_file)
    print(f"Wrote {len(all_entries)} entries to {csv_file}")


if __name__ == "__main__":
    main()
