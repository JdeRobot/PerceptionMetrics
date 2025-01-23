import argparse

from detectionmetrics.datasets.gaia import GaiaLiDARSegmentationDataset


def parse_args() -> argparse.Namespace:
    """Parse user input arguments

    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Parquet dataset file"
    )
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    GaiaLiDARSegmentationDataset(dataset_fname=args.dataset)


if __name__ == "__main__":
    main()
