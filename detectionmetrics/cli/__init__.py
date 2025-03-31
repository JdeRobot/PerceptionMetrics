from detectionmetrics import datasets
from detectionmetrics import models
from detectionmetrics.cli.evaluate import evaluate
from detectionmetrics.cli.computational_cost import computational_cost

REGISTRY = {
    "evaluate": evaluate,
    "computational_cost": computational_cost,
}


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
    images_dir,
    labels_dir,
    data_suffix,
    label_suffix,
    ontology,
    split,
):
    # Check if required data is available
    if dataset_format == "gaia":
        if dataset_fname is None:
            raise ValueError("--dataset is required for 'gaia' format")

    elif dataset_format in ["rellis3d", "wildscenes"]:
        if dataset_dir is None:
            raise ValueError(
                "--dataset_dir is required for 'rellis3d' and 'wildscenes' formats"
            )
        if split_dir is None:
            raise ValueError(
                "--split_dir is required for 'rellis3d' and 'wildscenes' formats"
            )

        if dataset_format == "rellis3d" and ontology is None:
            raise ValueError("--dataset_ontology is required for 'rellis3d' format")

    elif dataset_format in ["goose", "generic"]:
        if "train" in split and train_dataset_dir is None:
            raise ValueError(
                "--train_dataset_dir is required for 'train' split in 'goose' and 'generic' formats"
            )
        elif "val" in split and val_dataset_dir is None:
            raise ValueError(
                "--val_dataset_dir is required for 'val' split in 'goose' and 'generic' formats"
            )
        elif "test" in split and test_dataset_dir is None:
            raise ValueError(
                "--test_dataset_dir is required for 'test' split in 'goose' and 'generic' formats"
            )

        if dataset_format == "generic":
            if data_suffix is None:
                raise ValueError("--data_suffix is required for 'generic' format")
            if label_suffix is None:
                raise ValueError("--label_suffix is required for 'generic' format")
            if ontology is None:
                raise ValueError("--dataset_ontology is required for 'generic' format")

    elif dataset_format == "rugd":
        if images_dir is None:
            raise ValueError("--images_dir is required for 'rugd' format")
        if labels_dir is None:
            raise ValueError("--labels_dir is required for 'rugd' format")

    else:
        raise ValueError(f"Dataset format not supported: {dataset_format}")

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
    elif dataset_format == "rugd":
        dataset_args = {
            "images_dir": images_dir,
            "labels_dir": labels_dir,
            "ontology_fname": ontology,
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
