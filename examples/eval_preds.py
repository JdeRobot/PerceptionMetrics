import argparse
import os

from perceptionmetrics.datasets.gaia import GaiaImageSegmentationDataset


def parse_args() -> argparse.Namespace:
    """Parse user input arguments.

    :return: Parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Evaluate pre-computed prediction labels against a GT dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Parquet dataset file (GT)",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        required=True,
        help="Root directory containing prediction labels, organized in the "
        "same split/filename structure as the GT dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Name of the split to evaluate (default: test)",
    )
    parser.add_argument(
        "--out_fname",
        type=str,
        required=True,
        help="CSV file where the evaluation results will be stored",
    )
    parser.add_argument(
        "--ignored_classes",
        type=str,
        nargs="+",
        default=None,
        help="List of class names to ignore during evaluation",
    )
    parser.add_argument(
        "--results_per_sample",
        action="store_true",
        help="Store per-sample results as CSV files next to each prediction",
    )
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()

    dataset = GaiaImageSegmentationDataset(args.dataset)
    results = dataset.eval_preds(
        predictions_dir=args.predictions_dir,
        split=args.split,
        ignored_classes=args.ignored_classes,
        results_per_sample=args.results_per_sample,
    )

    os.makedirs(os.path.dirname(args.out_fname), exist_ok=True)
    results.to_csv(args.out_fname)
    print(f"Results saved to {args.out_fname}")


if __name__ == "__main__":
    main()
