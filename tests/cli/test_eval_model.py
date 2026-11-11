import os

import pandas as pd
import pytest
from click.testing import CliRunner

from perceptionmetrics import cli
from perceptionmetrics.cli.eval_model import eval_model


class FakeModel:
    """Model whose evaluation returns a fixed metrics table"""

    def __init__(self, as_dict):
        self.as_dict = as_dict

    def eval(self, dataset, **kwargs):
        """Return the metrics table the way each task does"""
        metrics_df = pd.DataFrame({"iou": [0.5]})
        return {"metrics_df": metrics_df} if self.as_dict else metrics_df


@pytest.fixture
def invoke(tmp_path, monkeypatch):
    """Run ``eval_model`` with a fake model and dataset and the given extra options"""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    for name in ("model.pt", "ontology.json", "cfg.json"):
        (input_dir / name).write_text("{}")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    monkeypatch.setattr(cli, "get_dataset", lambda *args, **kwargs: None)

    def _invoke(task, extra_args=(), as_dict=False):
        """Invoke the command for ``task`` and return the click result"""
        monkeypatch.setattr(
            cli, "get_model", lambda *args, **kwargs: FakeModel(as_dict)
        )
        return CliRunner().invoke(
            eval_model,
            [
                task,
                "image",
                "--model",
                str(input_dir / "model.pt"),
                "--model_ontology",
                str(input_dir / "ontology.json"),
                "--model_cfg",
                str(input_dir / "cfg.json"),
                *extra_args,
            ],
        )

    return _invoke, work_dir


@pytest.mark.parametrize(
    "task, as_dict", [("segmentation", False), ("detection", True)]
)
def test_eval_model_without_out_fname_does_not_write_results(invoke, task, as_dict):
    """``--out_fname`` is optional (e.g. when only ``--predictions_outdir`` is
    wanted): the evaluation must finish and no results file must be written"""
    run, work_dir = invoke

    result = run(task, as_dict=as_dict)

    assert result.exit_code == 0, result.output
    assert os.listdir(work_dir) == []


@pytest.mark.parametrize(
    "task, as_dict", [("segmentation", False), ("detection", True)]
)
def test_eval_model_writes_results_to_out_fname(invoke, tmp_path, task, as_dict):
    """The results table is written to ``--out_fname``, creating missing folders"""
    run, _ = invoke
    out_fname = tmp_path / "new_dir" / "results.csv"

    result = run(task, ["--out_fname", str(out_fname)], as_dict=as_dict)

    assert result.exit_code == 0, result.output
    assert pd.read_csv(out_fname, index_col=0)["iou"].tolist() == [0.5]
