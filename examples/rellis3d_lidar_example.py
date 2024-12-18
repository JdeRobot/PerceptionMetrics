import argparse

from detectionmetrics.datasets.rellis3d import Rellis3DLiDARSegmentationDataset


def parse_args() -> argparse.Namespace:
    """Parse user input arguments

    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory where dataset points and labels are stored",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        required=True,
        help="Directory where .lst files defining the dataset split are stored",
    )
    parser.add_argument(
        "--ontology_fname",
        type=str,
        required=True,
        help="YAML file containing dataset ontology",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory where dataset will be stored in common format",
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    dataset = Rellis3DLiDARSegmentationDataset(
        dataset_dir=args.dataset_dir,
        split_dir=args.split_dir,
        ontology_fname=args.ontology_fname,
    )
    dataset.export(args.outdir)


if __name__ == "__main__":
    main()
