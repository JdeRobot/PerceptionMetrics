import argparse

from perceptionmetrics.datasets.generic import GenericImageSegmentationDataset


def parse_args() -> argparse.Namespace:
    """Parse user input arguments

    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_suffix",
        type=str,
        required=True,
        help="File suffix to be used to filter data",
    )
    parser.add_argument(
        "--label_suffix",
        type=str,
        required=True,
        help="File suffix to be used to filter labels",
    )
    parser.add_argument(
        "--ontology_fname",
        type=str,
        required=True,
        help="JSON file containing either a list of classes or a dictionary with class "
        "names as keys and class indexes + rgb as values",
    )
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
        "--outdir",
        type=str,
        required=True,
        help="Directory where dataset will be stored in common format",
    )
    parser.add_argument(
        "--resize",
        type=str,
        required=False,
        help="Resize images to a specific size (e.g. 512x512)",
    )

    args = parser.parse_args()
    args.resize = tuple(map(int, args.resize.split("x"))) if args.resize else None
    return args


def main():
    """Main function"""
    args = parse_args()

    dataset = GenericImageSegmentationDataset(
        data_suffix=args.data_suffix,
        label_suffix=args.label_suffix,
        ontology_fname=args.ontology_fname,
        train_dataset_dir=args.train_dataset_dir,
        val_dataset_dir=args.val_dataset_dir,
        test_dataset_dir=args.test_dataset_dir,
    )
    dataset.export(args.outdir, resize=args.resize)


if __name__ == "__main__":
    main()
