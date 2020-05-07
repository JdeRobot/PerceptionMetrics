import sys
print("Python version")
print (sys.version)
print("Version info.")
print (sys.version_info)
import torch
from torchvision import transforms
import torchvision.models as models
from torch.autograd import Variable
from PIL import Image
import time

class PyTorchDetector:
    def __init__(self, patch_to_ckpt):
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = "cpu"
        #checkpoint = torch.load(checkpoint)
        self.model = self.model.to(device)
        self.model.eval()

    def run_inference_for_single_image(self, image):
        print('RUN INFERENCE')
        self.model.eval()
        detections = self.model([image])
        output_dict = {}
        output_dict['num_detections'] = len(detections[0]['labels'])
        output_dict['detection_classes'] = list(detections[0]['labels'].cpu().numpy())
        output_dict['detection_boxes'] = [[(i[0], i[1]), (i[2], i[3])] for i in list(detections[0]['boxes'].detach().cpu().numpy())]
        output_dict['detection_scores'] = list(detections[0]['scores'].detach().cpu().numpy())
        return output_dict

    def detect(self, img, threshold):
        print('DETECT')
        img_size=416
        Tensor = torch.cuda.FloatTensor

        ratio = min(img_size/img.shape[0], img_size/img.shape[1])
        imw = round(img.shape[0] * ratio)
        imh = round(img.shape[1] * ratio)
        img_transforms=transforms.Compose([transforms.ToTensor(),])
        image_tensor = img_transforms(img)
        input_img = Variable(image_tensor.type(Tensor))
        print('Starting inference')
        start_time = time.time()
        output_dict = self.run_inference_for_single_image(input_img)
        print("Inference Time: " + str(time.time() - start_time) + " seconds")
        
        pred_t = [output_dict['detection_scores'].index(x) for x in output_dict['detection_scores'] if x > threshold][-1]
        pred_boxes = output_dict['detection_boxes'][:pred_t+1]
        
        pred_class = output_dict['detection_classes'][:pred_t+1]
        new_dict = {}
        new_dict['detection_scores'] = pred_t
        new_dict['detection_boxes'] = pred_boxes
        new_dict['detection_classes'] = pred_class
        new_dict['num_detections'] = new_dict['detection_scores']
        print('----- DETECTION SCORES -----')
        print(new_dict['detection_scores'])
        print('------ DETECTION BOXES -----')
        print(new_dict['detection_boxes'])
        print('------ DETECITON CLASSES ------')
        print(new_dict['detection_classes'])
        print('------ DETECTION SCORES ------')
        print(new_dict['detection_scores'])
        
        return new_dict

