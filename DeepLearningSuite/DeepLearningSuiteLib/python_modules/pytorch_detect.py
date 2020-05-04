import torch
from torchvision import transforms
from torch.autograd import Variable

class PyTorchDetector:
    def __init__(self, patch_to_ckpt):
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #checkpoint = torch.load(checkpoint)
        model = model.to(device)
        model.eval()

    def run_inference_for_single_image(self, image):
        detections = model(image)
        output_dict = {}
        output_dict['num_detections'] = len(detections[0]['labels'])
        output_dict['detection_classes'] = list(detections[0]['labels'].numpy())
        output_dict['detection_boxes'] = [[(i[0], i[1]), (i[2], i[3])] for i in list(detections[0]['boxes'].detach().numpy())]
        output_dict['detection_scores'] = list(detections[0]['scores'].detach().numpy())

        '''
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        '''


        #predicted_locs, predicted_scores = model(image)
        #det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
        #                                                                               min_score=0.01, max_overlap=0.45,
        #                                                                               top_k=200)

        #boxes = [b.to(device) for b in boxes]
        #labels = [l.to(device) for l in labels]
        #difficulties = [d.to(device) for d in difficulties]
        
        return output_dict

    def detect(self, img, threshold):
        img_path = '/home/docker/Projects/DetectionSuite/datasets/coco/oneval2014/COCO_val2014_000000397133.jpg'
        img = Image.open(img_path)
        Tensor = torch.cuda.FloatTensor

        ratio = min(img_size/img.size[0], img_size/img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        print(str(imw) + ' ' +  str(imh))
        img_transforms=transforms.Compose([transforms.Resize((imh,imw)),transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)), (128,128,128)),transforms.ToTensor(),])

        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_img = Variable(image_tensor.type(Tensor))
        print('Starting inference')
        start_time = time.time()
        output_dict = self.run_inference_for_single_image(input_img)
        print("Inference Time: " + str(time.time() - start_time) + " seconds")

        pred_t = [output_dict['detection_scores'.index(x) for x in output_dict['detection_scores' if x > threshold][-1] # Get list of index with score greater than threshold.
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        new_dict = {}
        new_dict['detection_scores'] = pred_t
        new_dict['detection_boxes'] = pred_boxes
        new_dict['detection_classes'] = pred_class
        new_dict['num_detections'] = len(new_dict['detection_scores'])

        '''
        new_dict['detection_scores'] = output_dict['detection_scores'][output_dict['detection_scores']>=threshold]
        new_dict['detection_boxes'] = output_dict['detection_boxes'][0:len(new_dict['detection_scores']), :]
        new_dict['detection_classes'] = output_dict['detection_classes'][0:len(new_dict['detection_scores'])]
        new_dict['num_detections'] = len(new_dict['detection_scores'])

        if 'detection_masks' in output_dict:
            new_dict['detection_masks'] = output_dict['detection_masks'][0:len(new_dict['detection_scores']), :]
        '''
        return new_dict



