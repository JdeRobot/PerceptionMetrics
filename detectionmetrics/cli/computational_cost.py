import click

from detectionmetrics import cli
from detectionmetrics.utils.io import read_json


@click.command(name="computational_cost", help="Estimate model computational cost")
@click.argument("task", type=click.Choice(["segmentation"], case_sensitive=False))
@click.argument(
    "input_type", type=click.Choice(["image", "lidar"], case_sensitive=False)
)
# model
@click.option(
    "--model_format",
    type=click.Choice(
        ["torch", "tensorflow", "tensorflow_explicit"], case_sensitive=False
    ),
    show_default=True,
    default="torch",
    help="Trained model format",
)
@click.option(
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Trained model filename (TorchScript) or directory (TensorFlow SavedModel)",
)
@click.option(
    "--model_ontology",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="JSON file containing model output ontology",
)
@click.option(
    "--model_cfg",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="JSON file with model configuration (norm. parameters, image size, etc.)",
)
@click.option(
    "--image_size",
    type=(int, int),
    required=False,
    help="Dummy image size used for computational cost estimation",
)
# output
@click.option(
    "--out_fname",
    type=click.Path(writable=True),
    help="CSV file where the computational cost estimation results will be stored",
)
def computational_cost(
    task,
    input_type,
    model_format,
    model,
    model_ontology,
    model_cfg,
    image_size,
    out_fname,
):
    """Estimate model computational cost"""

    if image_size is None:
        parsed_model_cfg = read_json(model_cfg)
        if "image_size" in parsed_model_cfg:
            image_size = parsed_model_cfg["image_size"]
        else:
            raise ValueError(
                "Image size must be provided either as an argument or in the model configuration file"
            )

    model = cli.get_model(
        task, input_type, model_format, model, model_ontology, model_cfg
    )
    results = model.get_computational_cost(image_size)
    results.to_csv(out_fname)

    return results
