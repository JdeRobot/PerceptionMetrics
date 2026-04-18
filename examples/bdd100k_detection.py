import argparse

from perceptionmetrics.datasets.bdd100k import BDD100KDetectionDataset


def parse_args() -> argparse.Namespace:
    """Parse user input arguments

    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Root directory containing train/ and val/ image folders",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        required=True,
        help="Root directory containing train/ and val/ label JSON folders",
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    dataset = BDD100KDetectionDataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
    )

    print(f"Total samples: {len(dataset.dataset)}")
    if not dataset.dataset.empty:
        print(f"Splits: {dataset.dataset['split'].value_counts().to_dict()}")
        print(f"Ontology classes: {list(dataset.ontology.keys())}")


if __name__ == "__main__":
    main()
