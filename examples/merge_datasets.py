import argparse

from perceptionmetrics.datasets.gaia import GaiaImageSegmentationDataset, GaiaLiDARSegmentationDataset


def parse_args() -> argparse.Namespace:
    """Parse user input arguments

    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        nargs="+",
        help="Parquet dataset files",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory where merged dataset will be stored",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["image", "lidar"],
        required=True,
        help="Type of datasets to merge",
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    if args.dataset_type == "image":
        dataset_class = GaiaImageSegmentationDataset
    elif args.dataset_type == "lidar":
        dataset_class = GaiaLiDARSegmentationDataset
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    datasets = [dataset_class(fname) for fname in args.datasets]
    main_dataset = datasets[0]
    for extra_dataset in datasets[1:]:
        main_dataset.append(extra_dataset)
    main_dataset.export(args.outdir)


if __name__ == "__main__":
    main()
