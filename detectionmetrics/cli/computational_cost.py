import click

from detectionmetrics import cli


@click.command(name="computational_cost", help="Estimate model computational cost")
@click.argument("task", type=click.Choice(["segmentation"], case_sensitive=False))
@click.argument(
    "input_type", type=click.Choice(["image", "lidar"], case_sensitive=False)
)
# model
@click.option(
    "--model_format",
    type=click.Choice(["torch", "tensorflow"], case_sensitive=False),
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
    nargs=2,
    type=int,
    required=False,
    help="Dummy image size. Should be provided as two integers: width height",
)
@click.option(
    "--point_cloud_range",
    nargs=6,
    type=int,
    required=False,
    help="Dummy point cloud range (meters). Should be provided as six integers: x_min y_min z_min x_max y_max z_max",
)
@click.option(
    "--num_points",
    type=int,
    required=False,
    help="Number of points for the dummy point cloud (uniformly sampled)",
)
@click.option(
    "--has_intensity",
    is_flag=True,
    default=False,
    help="Whether the dummy point cloud has intensity values",
)
# output
@click.option(
    "--out_fname",
    type=click.Path(writable=True),
    required=True,
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
    point_cloud_range,
    num_points,
    has_intensity,
    out_fname,
):
    """Estimate model computational cost"""
    if input_type == "image":
        if image_size is None:
            raise ValueError("Image size must be provided for image models")
        if point_cloud_range is not None or num_points is not None:
            raise ValueError(
                "Point cloud range and number of points cannot be provided for image models"
            )
        if has_intensity:
            raise ValueError("Intensity flag cannot be set for image models")
        params = {"image_size": image_size}
    elif input_type == "lidar":
        if point_cloud_range is None or num_points is None:
            raise ValueError(
                "Point cloud range and number of points must be provided for lidar models"
            )
        if image_size is not None:
            raise ValueError("Image size cannot be provided for lidar models")

        params = {
            "point_cloud_range": point_cloud_range,
            "num_points": num_points,
            "has_intensity": has_intensity,
        }
    else:
        raise ValueError(f"Unknown input type: {input_type}")

    model = cli.get_model(
        task, input_type, model_format, model, model_ontology, model_cfg
    )
    results = model.get_computational_cost(**params)
    results.to_csv(out_fname)

    return results


if __name__ == "__main__":
    computational_cost()
