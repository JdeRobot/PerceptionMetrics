import argparse

from perceptionmetrics.models.tf_segmentation import TensorflowImageSegmentationModel


def parse_args() -> argparse.Namespace:
    """Parse user input arguments

    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Tensorflow model in SavedModel format"
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
        "--image_size",
        type=int,
        nargs=2,
        default=(500, 500),
        help="Dummy image size. Should be provided as two integers: width height",
    )
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    model = TensorflowImageSegmentationModel(args.model, args.model_cfg, args.ontology)
    computational_cost = model.get_computational_cost(args.image_size)
    if hasattr(computational_cost, "iloc"):
        computational_cost = computational_cost.iloc[0].to_dict()

    print("--- Computational cost ---")
    print(f"Input shape: {computational_cost['input_shape']}")
    if computational_cost["n_params"] is not None:
        print(f"Number of parameters: {computational_cost['n_params'] / 1e6:.2f} M")
    if computational_cost["inference_time_s"] is not None:
        print(f"Inference time: {computational_cost['inference_time_s'] * 1000:.2f} ms")
    if computational_cost["size_mb"] is not None:
        print(f"Model size: {computational_cost['size_mb']:.2f} MB")


if __name__ == "__main__":
    main()
