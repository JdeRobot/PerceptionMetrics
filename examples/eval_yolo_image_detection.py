import argparse
import os

from perceptionmetrics.datasets.yolo import YOLODataset
from perceptionmetrics.models.torch_detection import TorchImageDetectionModel


def parse_args() -> argparse.Namespace:
    """Parse user input arguments

    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Evaluate and Visualize YOLO detection models."
    )

    parser.add_argument(
        "--model", type=str, required=True, help="Scripted pytorch model"
    )
    parser.add_argument(
        "--ontology",
        type=str,
        required=True,
        help="JSON file containing model output ontology",
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        required=True,
        help="JSON file with model configuration",
    )
    parser.add_argument(
        "--dataset_fname", type=str, required=True, help="YOLO-like YAML dataset file"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing the dataset images",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Name of the split to be evaluated",
    )
    parser.add_argument(
        "--metrics_fname",
        type=str,
        required=True,
        help="CSV file where the evaluation results will be stored",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=False,
        help="Directory where the evaluation results and visualizations will be stored. Only used if 'save_visualizations' or 'results_per_sample' are True",
    )
    parser.add_argument(
        "--results_per_sample",
        action="store_true",
        help="Save results per sample",
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save visualizations for each sample",
    )

    args = parser.parse_args()

    if args.save_visualizations or args.results_per_sample:
        if args.outdir is None:
            raise ValueError(
                "'outdir' must be specified when 'save_visualizations' or 'results_per_sample' are True"
            )
    else:
        args.outdir = None

    return args


def main():
    """Main function"""
    args = parse_args()

    # Load image detection model
    model = TorchImageDetectionModel(args.model, args.model_cfg, args.ontology)

    # Load dataset for evaluation
    dataset = YOLODataset(args.dataset_fname, args.dataset_dir)

    # Evaluation loop
    results = model.eval(
        dataset,
        split=args.split,
        predictions_outdir=args.outdir,
        results_per_sample=args.results_per_sample,
        save_visualizations=args.save_visualizations,
    )

    # Store evaluation results
    metrics_df = results["metrics_df"]
    os.makedirs(os.path.dirname(args.metrics_fname), exist_ok=True)
    metrics_df.to_csv(args.metrics_fname)
    print(f"Evaluation results saved to {args.metrics_fname}")


if __name__ == "__main__":
    main()
