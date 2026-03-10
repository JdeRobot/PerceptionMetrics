from perceptionmetrics.models.segmentation import ImageSegmentationModel


class OnnxImageSegmentationModel(ImageSegmentationModel):
    def __init__(self, model, model_type, model_cfg, ontology_fname, model_fname):
        super().__init__(
            model=model,
            model_type=model_type,
            model_cfg=model_cfg,
            ontology_fname=ontology_fname,
            model_fname=model_fname,
        )