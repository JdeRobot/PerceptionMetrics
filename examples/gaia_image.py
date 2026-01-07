import argparse
import json

from perceptionmetrics.datasets.gaia import GaiaImageSegmentationDataset


def parse_args() -> argparse.Namespace:
    """Parse user input arguments

    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Parquet dataset file"
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
        help="Directory where dataset will be exported to",
    )
    parser.add_argument(
        "--resize",
        type=str,
        required=False,
        help="Resize images to a specific size (e.g. 512x512)",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=False,
        help="What split to export (train, val, test). Export all splits if not specified",
    )

    args = parser.parse_args()
    args.resize = tuple(map(int, args.resize.split("x"))) if args.resize else None
    return args


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

    dataset = GaiaImageSegmentationDataset(dataset_fname=args.dataset)
    if args.split:
        dataset.dataset = dataset.dataset[dataset.dataset["split"] == args.split]
        dataset.has_label_count = False

    dataset.export(
        outdir=args.outdir,
        resize=args.resize,
        new_ontology=new_ontology,
        ontology_translation=ontology_translation,
    )


if __name__ == "__main__":
    main()
