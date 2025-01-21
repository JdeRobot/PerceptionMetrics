from detectionmetrics.models.model import ImageSegmentationModel


class OnnxImageSegmentationModel(ImageSegmentationModel):
    def __init__(self, model, model_type, ontology_fname, model_cfg):
        super().__init__(model, model_type, ontology_fname, model_cfg)

    def inference(self, image):
        return super().inference(image)

    def eval(self, dataset, batch_size = 1, split = "all", ontology_translation = None):
        return super().eval(dataset, batch_size, split, ontology_translation)
