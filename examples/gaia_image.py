import argparse

from detectionmetrics.datasets.gaia import GaiaImageSegmentationDataset


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

    dataset = GaiaImageSegmentationDataset(dataset_fname=args.dataset)
    if args.split:
        dataset.dataset = dataset.dataset[dataset.dataset["split"] == args.split]
        dataset.has_label_count = False
    dataset.export(outdir=args.outdir, resize=args.resize)


if __name__ == "__main__":
    main()
