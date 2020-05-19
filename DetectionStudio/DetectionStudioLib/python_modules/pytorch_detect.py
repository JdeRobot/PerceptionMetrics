import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import time
import yaml
import importlib
import sys
import os

class PyTorchDetector:
    def __init__(self, patch_to_ckpt, configuration_file):
        with open(configuration_file, 'r') as stream:
            data_loaded = yaml.safe_load(stream)
            model_path = data_loaded['modelPath']
            model_name = data_loaded['modelName']
            import_name = data_loaded['importName']
            model_parameters = data_loaded['modelParameters']
        try:
            sys.path.append(os.path.dirname(model_path))
        except:
            print('Model path undefined')
        sys.path.append(os.path.dirname(model_path))
        models = importlib.import_module(import_name)
        # Model parameters are converted to actual Python variables
        variables = model_parameters.split(',')
        for i, var in enumerate(variables):
            name = 'variable'+str(i)
            try:
                value = int(var)
            except Exception as e:
                try:
                    value = str(var)
                except Exception as e:
                    value = bool(var)
            setattr(self, name, value)
        # The number of parameters modifies the way the function gets called
        self.model = eval(self.get_model_function(len(variables)))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()
        
    def get_model_function(self, x):
        return {
            1: 'getattr(models, model_name)(self.variable0)',
            2: 'getattr(models, model_name)(self.variable0, self.variable1)',
            3: 'getattr(models, model_name)(self.variable0, self.variable1, self.variable2)',
            4: 'getattr(models, model_name)(self.variable0, self.variable1, self.variable2, self.variable3)',
            5: 'getattr(models, model_name)(self.variable0, self.variable1, self.variable2, self.variable3, self.variable4)',
            6: 'getattr(models, model_name)(self.variable0, self.variable1, self.variable2, self.variable3, self.variable4, self.variable5)',
            7: 'getattr(models, model_name)(self.variable0, self.variable1, self.variable2, self.variable3, self.variable4, self.variable5, self.variable6)',
            8: 'getattr(models, model_name)(self.variable0, self.variable1, self.variable2, self.variable3, self.variable4, self.variable5, self.variable6, self.variable7)',
        }[x]


    def run_inference_for_single_image(self, image):
        self.model.eval()
        print('Starting inference')
        try:
            # Try with image tensor as list
            start_time = time.time()
            detections = self.model([image])
            print("Inference Time: " + str(time.time() - start_time) + " seconds")
        except Exception as e:
            # Try with image tensor alone
            start_time = time.time()
            image = image.unsqueeze(1)
            detections = self.model(image)
            print("Inference Time: " + str(time.time() - start_time) + " seconds")
        output_dict = {}
        output_dict['num_detections'] = len(detections[0]['labels'])
        output_dict['detection_classes'] = detections[0]['labels'].cpu().numpy()
        output_dict['detection_boxes'] = detections[0]['boxes'].detach().cpu().numpy()
        output_dict['detection_scores'] = detections[0]['scores'].detach().cpu().numpy()
        return output_dict

    def detect(self, img, threshold): 
        Tensor = torch.cuda.FloatTensor
        img_transforms=transforms.Compose([transforms.ToTensor(),])
        image_tensor = img_transforms(img)
        input_img = Variable(image_tensor.type(Tensor))
        output_dict = self.run_inference_for_single_image(input_img)        
        
        new_dict = {}
        new_dict['detection_scores'] = output_dict['detection_scores'][output_dict['detection_scores']>=threshold]
        new_dict['detection_boxes'] = output_dict['detection_boxes'][0:len(new_dict['detection_scores']), :]
        new_dict['detection_boxes'] = [[i[0]/img.shape[1], i[1]/img.shape[0], i[2]/img.shape[1], i[3]/img.shape[0]] for i in list(new_dict['detection_boxes'])] 
        new_dict['detection_boxes'] = np.float32(new_dict['detection_boxes'])
        new_dict['detection_classes'] = output_dict['detection_classes'][0:len(new_dict['detection_scores'])]
        new_dict['detection_classes'] = np.int8(new_dict['detection_classes'])
        new_dict['num_detections'] = len(new_dict['detection_scores'])

        return new_dict

