import argparse
import os


from perceptionmetrics.datasets.gaia import GaiaLiDARSegmentationDataset
from perceptionmetrics.models.torch_segmentation import TorchLiDARSegmentationModel
import perceptionmetrics.utils.conversion as uc
import perceptionmetrics.utils.lidar as ul


def parse_args() -> argparse.Namespace:
    """Parse user input arguments

    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
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
        help="JSON file withm model configuration (sampling, input format, etc.)",
    )
    parser.add_argument(
        "--point_cloud",
        type=str,
        required=False,
        help="Point cloud that will be segmented (in SemanticKITTI format)",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Parquet dataset file"
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Name of the split to be evaluated",
    )
    parser.add_argument(
        "--out_fname",
        type=str,
        required=True,
        help="CSV file where the evaluation results will be stored",
    )
    parser.add_argument(
        "--ontology_translation",
        type=str,
        required=False,
        help="JSON file containing translation between dataset and model classes",
    )
    parser.add_argument(
        "--translation_direction",
        type=str,
        choices=["dataset_to_model", "model_to_dataset"],
        default="dataset_to_model",
        help="Direction of the ontology translation",
    )
    parser.add_argument(
        "--predictions_outdir",
        type=str,
        required=False,
        help="Directory where the predictions will be stored. If not provided, "
        "predictions will not be saved",
    )
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    model = TorchLiDARSegmentationModel(args.model, args.model_cfg, args.ontology)
    dataset = GaiaLiDARSegmentationDataset(args.dataset)

    if args.point_cloud is not None:
        result = model.predict(args.point_cloud)
        lut = uc.ontology_to_rgb_lut(model.ontology)
        colors = lut[result] / 255.0
        point_cloud = dataset.read_points(args.point_cloud)
        ul.view_point_cloud(point_cloud[:, :3], colors)

    results = model.eval(
        dataset,
        split=args.split,
        ontology_translation=args.ontology_translation,
        translation_direction=args.translation_direction,
        predictions_outdir=args.predictions_outdir,
        results_per_sample=args.predictions_outdir is not None,
    )
    os.makedirs(os.path.dirname(args.out_fname), exist_ok=True)
    results.to_csv(args.out_fname)


if __name__ == "__main__":
    main()
