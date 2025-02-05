import argparse

from detectionmetrics.datasets.goose import GOOSEImageSegmentationDataset


def parse_args() -> argparse.Namespace:
    """Parse user input arguments

    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dataset_dir",
        type=str,
        required=True,
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

    dataset = GOOSEImageSegmentationDataset(
        train_dataset_dir=args.train_dataset_dir,
        val_dataset_dir=args.val_dataset_dir,
        test_dataset_dir=args.test_dataset_dir,
    )
    dataset.export(args.outdir, resize=args.resize)


if __name__ == "__main__":
    main()
