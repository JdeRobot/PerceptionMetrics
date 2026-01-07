import argparse
import os
import shutil
import subprocess

from perceptionmetrics.datasets.gaia import GaiaLiDARSegmentationDataset
from perceptionmetrics.models.torch_segmentation import TorchLiDARSegmentationModel
import perceptionmetrics.utils.conversion as uc
import perceptionmetrics.utils.lidar as ul
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parse user input arguments

    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=False, help="Scripted pytorch model"
    )
    parser.add_argument(
        "--ontology",
        type=str,
        required=False,
        help="JSON file containing model output ontology",
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        required=False,
        help="JSON file withm model configuration (sampling, input format, etc.)",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Parquet dataset file"
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=False,
        help="Scene to be processed",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=False,
        help="Name of the split to be evaluated",
    )
    parser.add_argument(
        "--out_fname",
        type=str,
        required=True,
        help="File name for the output video",
    )
    parser.add_argument(
        "--ontology_translation",
        type=str,
        required=False,
        help="JSON file containing translation between dataset and model classes",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the output video",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="1920x1080",
        help="Resolution of the output video (e.g. 1920x1080)",
    )

    args = parser.parse_args()
    args.resolution = tuple(int(d) for d in args.resolution.split("x"))

    return args


def main():
    """Main function"""
    args = parse_args()

    # Init model if required, otherwise we will simply render the ground truth labels
    model = None
    if args.model is not None:
        assert args.model_cfg is not None, "Model configuration file is required"
        assert args.ontology is not None, "Ontology file is required"
        model = TorchLiDARSegmentationModel(args.model, args.model_cfg, args.ontology)

    # Init dataset
    dataset = GaiaLiDARSegmentationDataset(args.dataset)
    dataset.make_fname_global()

    # Filter dataset if required
    if args.split is not None:
        dataset.dataset = dataset.dataset[dataset.dataset["split"] == args.split]
    if args.scene is not None:
        dataset.dataset = dataset.dataset[dataset.dataset["scene"] == args.scene]

    # Create output and temporary directory
    tmp_dir, _ = os.path.splitext(args.out_fname)
    tmp_dir += "_tmp"
    os.makedirs(os.path.dirname(args.out_fname), exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # Render point clouds
    pbar = tqdm(enumerate(dataset.dataset.iterrows()), total=len(dataset.dataset))
    for idx, (sample_name, sample_data) in pbar:
        pbar.set_description(f"Processing sample {sample_name}")
        point_cloud = dataset.read_points(sample_data["points"])

        if model is not None:
            label = model.predict(point_cloud)
            lut = uc.ontology_to_rgb_lut(model.ontology)
        else:
            label = dataset.read_label(sample_data["label"])
            lut = uc.ontology_to_rgb_lut(dataset.ontology)
        colors = lut[label] / 255.0

        rendered_image = ul.render_point_cloud(
            point_cloud[:, :3], colors, resolution=args.resolution
        )
        rendered_image.save(os.path.join(tmp_dir, f"frame_{idx:06d}.png"))

    # Create video with ffmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-framerate",
        str(args.fps),
        "-i",
        os.path.join(tmp_dir, "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        args.out_fname,
    ]
    subprocess.run(ffmpeg_command)

    # Remove temporary directory
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
