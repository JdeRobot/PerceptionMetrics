import argparse

from PIL import Image

from perceptionmetrics.datasets.gaia import GaiaImageSegmentationDataset
from perceptionmetrics.models.tensorflow import TensorflowImageSegmentationModel
import perceptionmetrics.utils.conversion as uc
import tensorflow as tf

from local import tensorflow_advanced_segmentation_models as tasm


def parse_args() -> argparse.Namespace:
    """Parse user input arguments

    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_weights", type=str, required=True, help="Tensorflow model weights in HDF5 format"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="TASM model name"
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
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    base_model, layers, _ = tasm.create_base_model(
        name="resnet50",
        weights="imagenet",
        height=320,
        width=320,
        include_top=False,
        pooling=None,
    )

    model_init_args = {
        "n_classes": 20,
        "base_model": base_model,
        "output_layers": layers,
        "backbone_trainable": False,
    }
    tasm_model = getattr(tasm, args.model_name)(**model_init_args)

    # Compile model (is it necessary to configure optimizer?)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-7)
    tasm_model.compile(
        optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    tasm_model.predict(tf.zeros((1, 320, 320, 3)))
    tasm_model.load_weights(args.model_weights)

    model = TensorflowImageSegmentationModel(tasm_model, args.model_cfg, args.ontology)
    dataset = GaiaImageSegmentationDataset(args.dataset)

    if args.image is not None:
        image = Image.open(args.image).convert("RGB")
        result = model.inference(image)
        result = uc.label_to_rgb(result, model.ontology)
        result.show()

    results = model.eval(
        dataset,
        split=args.split,
        ontology_translation=args.ontology_translation,
        predictions_outdir="local/data/tf_predictions",
        results_per_sample=True,
    )
    results.to_csv(args.out_fname)


if __name__ == "__main__":
    main()
