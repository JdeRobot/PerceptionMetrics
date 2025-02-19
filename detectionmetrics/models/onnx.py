from detectionmetrics.models.model import ImageSegmentationModel


class OnnxImageSegmentationModel(ImageSegmentationModel):
    def __init__(self, model, model_type, ontology_fname, model_cfg, model_fname):
        super().__init__(model, model_type, ontology_fname, model_cfg, model_fname)
