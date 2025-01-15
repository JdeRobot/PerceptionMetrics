import click

from detectionmetrics import datasets
from detectionmetrics import models


def get_model(task, input_type, model_format, model, ontology, model_cfg):
    # Init model from registry
    model_name = f"{model_format}_{input_type}_{task}"
    if model_name not in models.REGISTRY:
        raise ValueError(
            f"Model format not supported: {model_format}. "
            f"Must be one of {models.REGISTRY.keys()}",
        )
    return models.REGISTRY[model_name](model, model_cfg, ontology)


def get_dataset(
    task,
    input_type,
    dataset_format,
    dataset_fname,
    dataset_dir,
    split_dir,
    train_dataset_dir,
    val_dataset_dir,
    test_dataset_dir,
    data_suffix,
    label_suffix,
    ontology,
    split,
):
    # Check if required data is available
    if dataset_format == "gaia" and dataset_fname is None:
        raise ValueError("--dataset is required for 'gaia' format")

    elif dataset_format == "rellis3d":
        if dataset_dir is None:
            raise ValueError("--dataset_dir is required for 'rellis3d' format")
        if split_dir is None:
            raise ValueError("--split_dir is required for 'rellis3d' format")
        if ontology is None:
            raise ValueError("--dataset_ontology is required for 'rellis3d' format")

    elif dataset_format in ["goose", "generic"]:
        if split == "train" and train_dataset_dir is None:
            raise ValueError(
                "--train_dataset_dir is required for 'train' split in 'goose' format"
            )
        elif split == "val" and val_dataset_dir is None:
            raise ValueError(
                "--val_dataset_dir is required for 'val' split in 'goose' format"
            )
        elif split == "test" and test_dataset_dir is None:
            raise ValueError(
                "--test_dataset_dir is required for 'test' split in 'goose' format"
            )

        if dataset_format == "generic":
            if data_suffix is None:
                raise ValueError("--data_suffix is required for 'generic' format")
            if label_suffix is None:
                raise ValueError("--label_suffix is required for 'generic' format")
            if ontology is None:
                raise ValueError("--dataset_ontology is required for 'generic' format")

    # Get arguments to init dataset
    if dataset_format == "gaia":
        dataset_args = {"dataset_fname": dataset_fname}
    elif dataset_format == "rellis3d":
        dataset_args = {
            "dataset_dir": dataset_dir,
            "split_dir": split_dir,
            "ontology_fname": ontology,
        }
    elif dataset_format == "goose":
        dataset_args = {
            "train_dataset_dir": train_dataset_dir,
            "val_dataset_dir": val_dataset_dir,
            "test_dataset_dir": test_dataset_dir,
        }
    elif dataset_format == "generic":
        dataset_args = {
            "data_suffix": data_suffix,
            "label_suffix": label_suffix,
            "ontology_fname": ontology,
            "train_dataset_dir": train_dataset_dir,
            "val_dataset_dir": val_dataset_dir,
            "test_dataset_dir": test_dataset_dir,
        }
    else:
        raise ValueError(f"Dataset format not supported: {dataset_format}")

    # Init dataset from registry
    dataset_name = f"{dataset_format}_{input_type}_{task}"
    if dataset_name not in datasets.REGISTRY:
        raise ValueError(
            f"Dataset format not supported: {dataset_format}. "
            f"Must be one of {datasets.REGISTRY.keys()}",
        )
    return datasets.REGISTRY[dataset_name](**dataset_args)


@click.command(name="evaluate", help="Evaluate model on dataset")
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
# dataset
@click.option(
    "--dataset_format",
    type=click.Choice(["gaia", "rellis3d", "goose", "generic"], case_sensitive=False),
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
    help="Dataset directory (used for 'Rellis3D' format)",
)
@click.option(
    "--split_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory containing .lst split files (used for 'Rellis3D' format)",
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
    type=click.Choice(["train", "val", "test"], case_sensitive=False),
    show_default=True,
    default="test",
    help="Name of the split to be evaluated",
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
    required=True,
    help="CSV file where the evaluation results will be stored",
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
    data_suffix,
    label_suffix,
    dataset_ontology,
    split,
    ontology_translation,
    out_fname,
):
    """Evaluate model on dataset"""

    model = get_model(task, input_type, model_format, model, model_ontology, model_cfg)
    dataset = get_dataset(
        task,
        input_type,
        dataset_format,
        dataset_fname,
        dataset_dir,
        split_dir,
        train_dataset_dir,
        val_dataset_dir,
        test_dataset_dir,
        data_suffix,
        label_suffix,
        dataset_ontology,
        split,
    )

    results = model.eval(
        dataset,
        split=split,
        ontology_translation=ontology_translation,
    )
    results.to_csv(out_fname)

    return results


if __name__ == "__main__":
    evaluate()
