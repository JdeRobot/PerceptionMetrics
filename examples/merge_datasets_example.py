import argparse

from detectionmetrics.datasets.gaia import GaiaImageSegmentationDataset


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

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    datasets = [GaiaImageSegmentationDataset(fname) for fname in args.datasets]
    main_dataset = datasets[0]
    for extra_dataset in datasets[1:]:
        main_dataset.append(extra_dataset)
    main_dataset.export(args.outdir)


if __name__ == "__main__":
    main()
