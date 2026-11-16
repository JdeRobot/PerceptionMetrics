import json

import cv2
import numpy as np
import pandas as pd
import yaml
from click.testing import CliRunner

from perceptionmetrics.cli.batch import batch


def test_batch_eval_preds_stores_results_per_sample(tmp_path):
    """pm_batch eval_preds writes results when store_results_per_sample is set.

    :param tmp_path: Temporary directory provided by pytest
    :type tmp_path: pathlib.Path
    """
    # Tiny Gaia dataset with one label and a prediction that mirrors it
    label = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    for root in ("dataset", "preds"):
        (tmp_path / root / "test").mkdir(parents=True)
        cv2.imwrite(str(tmp_path / root / "test" / "label-a.png"), label)
    ontology = {"bg": {"idx": 0, "rgb": [0, 0, 0]}, "fg": {"idx": 1, "rgb": [1, 1, 1]}}
    (tmp_path / "dataset" / "ontology.json").write_text(json.dumps(ontology))
    samples = pd.DataFrame(
        {
            "image": ["test/image-a.png"],
            "label": ["test/label-a.png"],
            "split": ["test"],
        },
        index=["a"],
    )
    samples.to_parquet(tmp_path / "dataset" / "dataset.parquet")

    jobs_cfg = {
        "task": "segmentation",
        "input_type": "image",
        "id": "tiny",
        "predictions": {"id": "preds", "path": str(tmp_path / "preds")},
        "dataset": {
            "id": "gaia",
            "format": "gaia",
            "fname": str(tmp_path / "dataset" / "dataset.parquet"),
            "split": "test",
        },
        "outdir": str(tmp_path / "out"),
        "store_results_per_sample": True,
    }
    (tmp_path / "batch.yaml").write_text(yaml.safe_dump(jobs_cfg))

    result = CliRunner().invoke(batch, ["eval_preds", str(tmp_path / "batch.yaml")])

    assert "Error processing job" not in result.output, result.output
    assert (tmp_path / "out" / "results-tiny.csv").is_file()
    assert (tmp_path / "preds" / "test" / "a.csv").is_file()
