import argparse

from perceptionmetrics.datasets import RUGDImageSegmentationDataset


def parse_args() -> argparse.Namespace:
    """Parse user input arguments

    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory where train dataset split is stored",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        help="Directory where validation dataset split is stored",
    )
    parser.add_argument(
        "--ontology",
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

    dataset = RUGDImageSegmentationDataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        ontology_fname=args.ontology,
    )
    dataset.export(args.outdir, resize=args.resize)


if __name__ == "__main__":
    main()
