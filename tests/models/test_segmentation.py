import json
import warnings

import numpy as np
import pytest

from perceptionmetrics.models.segmentation import SegmentationModel


class StubSegmentationModel(SegmentationModel):
    """Minimal concrete model that skips file loading, to exercise ontology handling."""

    def __init__(self, ontology: dict):
        self.ontology = ontology
        self.model_cfg = {}

    def inference(self, data):
        raise NotImplementedError

    def predict(self, data, return_sample=False):
        raise NotImplementedError

    def eval(self, *args, **kwargs):
        raise NotImplementedError

    def get_computational_cost(self, runs=30, warm_up_runs=5):
        raise NotImplementedError


# To verify that matching ontologies need no conversion
def test_get_eval_lut_ontology_matching_indices():
    ontology = {"car": {"idx": 1}, "tree": {"idx": 2}}
    model = StubSegmentationModel(ontology)

    lut, eval_ontology = model.get_eval_lut_ontology(dict(ontology))

    assert lut is None
    assert eval_ontology == ontology


# To verify that shared class names with differing indices are converted with a warning
def test_get_eval_lut_ontology_differing_indices_warns():
    dataset_ontology = {"car": {"idx": 1}, "tree": {"idx": 3}}
    model = StubSegmentationModel({"car": {"idx": 0}, "tree": {"idx": 1}})

    with pytest.warns(UserWarning, match="different class indices"):
        lut, eval_ontology = model.get_eval_lut_ontology(dataset_ontology)

    assert np.array_equal(lut, [0, 0, 0, 1])  # dataset idx 1 -> 0, idx 3 -> 1
    assert eval_ontology == model.ontology


# To verify that the model_to_dataset direction reports metrics in the dataset ontology
def test_get_eval_lut_ontology_model_to_dataset():
    dataset_ontology = {"car": {"idx": 1}, "tree": {"idx": 3}}
    model = StubSegmentationModel({"car": {"idx": 0}, "tree": {"idx": 1}})

    with pytest.warns(UserWarning, match="different class indices"):
        lut, eval_ontology = model.get_eval_lut_ontology(
            dataset_ontology, translation_direction="model_to_dataset"
        )

    assert np.array_equal(lut, [1, 3])  # model idx 0 -> 1, idx 1 -> 3
    assert eval_ontology == dataset_ontology


# To verify that differing class names require an explicit translation
def test_get_eval_lut_ontology_differing_names_raises():
    dataset_ontology = {"car": {"idx": 1}, "grass": {"idx": 2}}
    model = StubSegmentationModel({"car": {"idx": 1}, "vegetation": {"idx": 2}})

    with pytest.raises(ValueError, match="do not share the same class names"):
        model.get_eval_lut_ontology(dataset_ontology)


# To verify that an explicit translation file maps differing class names without warning
def test_get_eval_lut_ontology_explicit_translation(tmp_path):
    dataset_ontology = {"car": {"idx": 1}, "grass": {"idx": 2}}
    model = StubSegmentationModel({"vehicle": {"idx": 10}, "vegetation": {"idx": 20}})

    translation_fname = tmp_path / "translation.json"
    translation_fname.write_text(
        json.dumps({"car": "vehicle", "grass": "vegetation"}), encoding="utf-8"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an explicit translation must not warn
        lut, eval_ontology = model.get_eval_lut_ontology(
            dataset_ontology, str(translation_fname)
        )

    assert np.array_equal(lut, [0, 10, 20])
    assert eval_ontology == model.ontology


# To verify that an unknown translation direction is rejected
def test_get_eval_lut_ontology_invalid_direction():
    ontology = {"car": {"idx": 1}}
    model = StubSegmentationModel(dict(ontology))

    with pytest.raises(ValueError, match="Invalid translation direction"):
        model.get_eval_lut_ontology(ontology, None, "sideways")
