import click

from detectionmetrics import cli


def parse_split(ctx, param, value):
    """Parse split argument"""
    splits = value.split(",")
    valid_splits = ["train", "val", "test"]
    if not all(split in valid_splits for split in splits):
        raise click.BadParameter(
            f"Split must be one of {valid_splits} or a comma-separated list of them",
            param_hint=value,
        )

    return splits


@click.command(name="evaluate", help="Evaluate model on dataset")
@click.argument("task", type=click.Choice(["segmentation"], case_sensitive=False))
@click.argument(
    "input_type", type=click.Choice(["image", "lidar"], case_sensitive=False)
)
# model
@click.option(
    "--model_format",
    type=click.Choice(
        ["torch", "tensorflow"], case_sensitive=False
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
# dataset
@click.option(
    "--dataset_format",
    type=click.Choice(
        ["gaia", "rellis3d", "goose", "generic", "rugd"], case_sensitive=False
    ),
    show_default=True,
    default="gaia",
    help="Dataset format",
)
@click.option(
    "--dataset_fname",
    type=click.Path(exists=True, dir_okay=False),
    help="Parquet dataset file",
)
@click.option(
    "--dataset_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Dataset directory (used for 'Rellis3D' and 'Wildscenes' formats)",
)
@click.option(
    "--split_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory containing .lst or .csv split files (used for 'Rellis3D' and 'Wildscenes' formats, respectively)",
)
@click.option(
    "--train_dataset_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Train dataset directory (used for 'GOOSE' and 'Generic' formats)",
)
@click.option(
    "--val_dataset_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Validation dataset directory (used for 'GOOSE' and 'Generic' formats)",
)
@click.option(
    "--test_dataset_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Test dataset directory (used for 'GOOSE' and 'Generic' formats)",
)
@click.option(
    "--images_dir",
    type=click.STRING,
    help="Directory containing data (used for 'RUGD' format)",
)
@click.option(
    "--labels_dir",
    type=click.STRING,
    help="Directory containing annotations (used for 'RUGD' format)",
)
@click.option(
    "--data_suffix",
    type=click.STRING,
    help="Data suffix to be used to filter data (used for 'Generic' format)",
)
@click.option(
    "--label_suffix",
    type=click.STRING,
    help="Label suffix to be used to filter labels (used for 'Generic' format)",
)
@click.option(
    "--dataset_ontology",
    type=click.Path(exists=True, dir_okay=False),
    help="JSON containing dataset ontology (used for 'Generic' and 'Rellis3D' formats)",
)
@click.option(
    "--split",
    show_default=True,
    default="test",
    callback=parse_split,
    help="Name of the split or splits separated by commas to be evaluated",
)
@click.option(
    "--ontology_translation",
    type=click.Path(exists=True, dir_okay=False),
    help="JSON file containing translation between dataset and model classes",
)
# output
@click.option(
    "--out_fname",
    type=click.Path(writable=True),
    help="CSV file where the evaluation results will be stored",
)
@click.option(
    "--predictions_outdir",
    type=click.Path(writable=True),
    help="Directory where predictions (images/points and CSV) per sample will be "
    "stored. If not provided, predictions per sample will not be saved",
)
def evaluate(
    task,
    input_type,
    model_format,
    model,
    model_ontology,
    model_cfg,
    dataset_format,
    dataset_fname,
    dataset_dir,
    split_dir,
    train_dataset_dir,
    val_dataset_dir,
    test_dataset_dir,
    images_dir,
    labels_dir,
    data_suffix,
    label_suffix,
    dataset_ontology,
    split,
    ontology_translation,
    out_fname,
    predictions_outdir,
):
    """Evaluate model on dataset"""
    if isinstance(split, str):  # if evaluate has been called directly
        split = parse_split(None, None, split)

    model = cli.get_model(
        task, input_type, model_format, model, model_ontology, model_cfg
    )
    dataset = cli.get_dataset(
        task,
        input_type,
        dataset_format,
        dataset_fname,
        dataset_dir,
        split_dir,
        train_dataset_dir,
        val_dataset_dir,
        test_dataset_dir,
        images_dir,
        labels_dir,
        data_suffix,
        label_suffix,
        dataset_ontology,
        split,
    )

    results = model.eval(
        dataset,
        split=split,
        ontology_translation=ontology_translation,
        predictions_outdir=predictions_outdir,
        results_per_sample=predictions_outdir is not None,
    )
    results.to_csv(out_fname)

    return results


if __name__ == "__main__":
    evaluate()
