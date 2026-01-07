import argparse
import os

from PIL import Image

from perceptionmetrics.datasets.gaia import GaiaImageSegmentationDataset
from perceptionmetrics.models.torch_segmentation import TorchImageSegmentationModel
import perceptionmetrics.utils.conversion as uc


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
        help="JSON file withm model configuration (norm. parameters, image size, etc.)",
    )
    parser.add_argument(
        "--image", type=str, required=False, help="Image that will be segmented"
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

    model = TorchImageSegmentationModel(args.model, args.model_cfg, args.ontology)
    dataset = GaiaImageSegmentationDataset(args.dataset)

    if args.image is not None:
        image = Image.open(args.image).convert("RGB")
        result = model.predict(image)
        result = uc.label_to_rgb(result, model.ontology)
        result.show()

    results = model.eval(
        dataset,
        split=args.split,
        ontology_translation=args.ontology_translation,
        predictions_outdir=args.predictions_outdir,
        results_per_sample=args.predictions_outdir is not None,
    )
    os.makedirs(os.path.dirname(args.out_fname), exist_ok=True)
    results.to_csv(args.out_fname)


if __name__ == "__main__":
    main()
