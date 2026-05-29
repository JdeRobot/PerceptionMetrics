import os

import click

from perceptionmetrics import cli


def parse_split(ctx, param, value):
    """Parse split argument."""
    splits = value.split(",")
    valid_splits = ["train", "val", "test"]
    if not all(split in valid_splits for split in splits):
        raise click.BadParameter(
            f"Split must be one of {valid_splits} or a comma-separated list of them",
            param_hint=value,
        )
    return splits


@click.command(
    name="eval_preds",
    help="Evaluate pre-computed predictions stored on disk against a GT dataset",
)
@click.argument(
    "task",
    type=click.Choice(["segmentation", "detection"], case_sensitive=False),
)
@click.argument(
    "input_type",
    type=click.Choice(["image", "lidar"], case_sensitive=False),
)
# predictions
@click.option(
    "--predictions_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Root directory containing prediction files, organized in the same "
    "split/filename structure as the GT dataset",
)
# dataset
@click.option(
    "--dataset_format",
    type=click.Choice(
        ["gaia", "rellis3d", "goose", "generic", "rugd", "coco"],
        case_sensitive=False,
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
    help="Dataset directory (used for 'Rellis3D', 'Wildscenes', and 'COCO' formats)",
)
@click.option(
    "--split_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory containing .lst or .csv split files (used for 'Rellis3D' "
    "and 'Wildscenes' formats, respectively)",
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
# ontology translation
@click.option(
    "--pred_ontology",
    type=click.Path(exists=True, dir_okay=False),
    help="JSON file containing the prediction ontology (only needed when it "
    "differs from the dataset ontology)",
)
@click.option(
    "--ontology_translation",
    type=click.Path(exists=True, dir_okay=False),
    help="JSON file containing translation between dataset and prediction ontologies",
)
@click.option(
    "--translation_direction",
    type=click.Choice(["dataset_to_model", "model_to_dataset"], case_sensitive=False),
    default="dataset_to_model",
    show_default=True,
    help="Direction of the ontology translation",
)
# ignored classes
@click.option(
    "--ignored_classes",
    type=click.STRING,
    multiple=True,
    help="Class name(s) to ignore during evaluation (repeat for multiple)",
)
# output
@click.option(
    "--out_fname",
    type=click.Path(writable=True),
    required=True,
    help="CSV file where the evaluation results will be stored",
)
@click.option(
    "--results_per_sample",
    is_flag=True,
    default=False,
    help="Store per-sample CSV results next to each prediction file",
)
def eval_preds(
    task,
    input_type,
    predictions_dir,
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
    pred_ontology,
    ontology_translation,
    translation_direction,
    ignored_classes,
    out_fname,
    results_per_sample,
):
    """Evaluate pre-computed predictions stored on disk against a GT dataset."""
    import perceptionmetrics.utils.io as uio

    if isinstance(split, str):  # if eval_preds has been called directly
        split = parse_split(None, None, split)

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

    # Build eval_preds keyword arguments
    eval_kwargs = {
        "predictions_dir": predictions_dir,
        "split": split,
        "translation_direction": translation_direction,
        "results_per_sample": results_per_sample,
    }

    if pred_ontology is not None:
        eval_kwargs["pred_ontology"] = uio.read_json(pred_ontology)

    if ontology_translation is not None:
        eval_kwargs["ontology_translation"] = uio.read_json(ontology_translation)

    if ignored_classes:
        eval_kwargs["ignored_classes"] = list(ignored_classes)

    results = dataset.eval_preds(**eval_kwargs)

    os.makedirs(os.path.dirname(os.path.abspath(out_fname)), exist_ok=True)
    results.to_csv(out_fname)

    return results


if __name__ == "__main__":
    eval_preds()
