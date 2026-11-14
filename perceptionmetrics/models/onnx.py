from perceptionmetrics.models.segmentation import ImageSegmentationModel
import onnxruntime as ort
import os


class OnnxImageSegmentationModel(ImageSegmentationModel):
    def __init__(self, model, model_type, model_cfg, ontology_fname, model_fname):

        #Get Device (CPU and GPU)
        providers = ort.get_available_provziders()

        if isinstance(model,str):
            assert os.path.isfile(model),"Onnx Model file not found"
            model_fname = model
        else:
            raise ValueError("Model must be a Onnx Model File name")    

        
        super().__init__(
            model=model,
            model_type=model_type,
            model_cfg=model_cfg,
            ontology_fname=ontology_fname,
            model_fname=model_fname,
        )

        self.model(ort.InferenceSession(model_fname,providers = providers))

        # Init transformations for input images, output labels, and GT labels
        self.transform_input = []
        self.transform_label = []

    
        

        

        