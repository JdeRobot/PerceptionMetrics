import argparse
import json

from perceptionmetrics.datasets.goose import GOOSELiDARSegmentationDataset


def parse_args() -> argparse.Namespace:
    """Parse user input arguments

    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dataset_dir",
        type=str,
        help="Directory where train dataset split is stored",
    )
    parser.add_argument(
        "--val_dataset_dir",
        type=str,
        help="Directory where validation dataset split is stored",
    )
    parser.add_argument(
        "--test_dataset_dir",
        type=str,
        help="Directory where test dataset split is stored",
    )
    parser.add_argument(
        "--new_ontology",
        type=str,
        help="New ontology JSON file name",
    )
    parser.add_argument(
        "--ontology_translation",
        type=str,
        help="Ontology translation JSON file name",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory where dataset will be stored in common format",
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    new_ontology, ontology_translation = None, None
    if args.new_ontology is not None:
        with open(args.new_ontology, "r", encoding="utf-8") as f:
            new_ontology = json.load(f)

    if args.ontology_translation is not None:
        with open(args.ontology_translation, "r", encoding="utf-8") as f:
            ontology_translation = json.load(f)

    dataset = GOOSELiDARSegmentationDataset(
        train_dataset_dir=args.train_dataset_dir,
        val_dataset_dir=args.val_dataset_dir,
        test_dataset_dir=args.test_dataset_dir,
    )
    dataset.export(
        args.outdir,
        new_ontology=new_ontology,
        ontology_translation=ontology_translation,
    )


if __name__ == "__main__":
    main()
