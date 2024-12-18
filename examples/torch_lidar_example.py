import argparse


from detectionmetrics.datasets.gaia import GaiaLiDARSegmentationDataset
from detectionmetrics.models.torch import TorchLiDARSegmentationModel
import detectionmetrics.utils.conversion as uc
import detectionmetrics.utils.lidar as ul


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
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    model = TorchLiDARSegmentationModel(args.model, args.model_cfg, args.ontology)
    dataset = GaiaLiDARSegmentationDataset(args.dataset)

    if args.point_cloud is not None:
        point_cloud = dataset.read_points(args.point_cloud)
        result = model.inference(point_cloud)
        lut = uc.ontology_to_rgb_lut(model.ontology)
        colors = lut[result] / 255.0
        ul.render_point_cloud(point_cloud[:, :3], colors)

    results = model.eval(
        dataset,
        split=args.split,
        ontology_translation=args.ontology_translation,
    )
    results.to_csv(args.out_fname)


if __name__ == "__main__":
    main()
