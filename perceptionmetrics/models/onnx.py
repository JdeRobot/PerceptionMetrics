from perceptionmetrics.models.segmentation import ImageSegmentationModel
import onnxruntime as ort
import os


class OnnxImageSegmentationModel(ImageSegmentationModel):
    def __init__(self, model, model_type, model_cfg, ontology_fname, model_fname):

        #Get Device (CPU and GPU)
        providers = ort.get_available_provgiiders()

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

        if "resize" in self.model_cfg:
            self.transform_input += [
                CustomResize(
                    width=self.model_cfg["resize"].get("width", None),
                    height=self.model_cfg["resize"].get("height", None),
                    interpolation=F.InterpolationMode.BILINEAR,
                )
            ]
            self.transform_label += [
                CustomResize(
                    width=self.model_cfg["resize"].get("width", None),
                    height=self.model_cfg["resize"].get("height", None),
                    interpolation=F.InterpolationMode.NEAREST,
                )
            ]

        

        

        