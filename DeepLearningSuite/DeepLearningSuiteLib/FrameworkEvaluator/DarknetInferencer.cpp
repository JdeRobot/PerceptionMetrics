//
// Created by frivas on 31/01/17.
//
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
// OpenCV
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <Common/Sample.h>
#include <DatasetConverters/ClassTypeGeneric.h>
#include "DarknetInferencer.h"
#include <glog/logging.h>



using namespace std;
using namespace cv;
using namespace dnn;


DarknetInferencer::DarknetInferencer(const std::string &netConfig, const std::string &netWeights,const std::string& classNamesFile): netConfig(netConfig),netWeights(netWeights) {
    this->classNamesFile=classNamesFile;
    
    
    
    //this->cnn = boost::shared_ptr<DarknetAPI>(new DarknetAPI((char*)this->netConfig.c_str(), (char*)this->netWeights.c_str()));
}

Sample DarknetInferencer::detectImp(const cv::Mat &image, double confThreshold) {
       printf("OpenCV: %s", cv::getBuildInformation().c_str());
 
	float nmsThreshold = 0.4; // Non-maximum suppression threshold
	std::vector<string> classes = {};
string classesFile = "/home/docker/darknet/data/coco.names";
        ifstream ifs(classesFile.c_str());
        string line;
        while (getline(ifs, line)) classes.push_back(line);

        //Give the configuration and weight files for the model
        string modelConfiguration = "/home/docker/Projects/DetectionSuite/datasets/cfg/yolov3.cfg";
        string modelWeights = "/home/docker/Projects/DetectionSuite/datasets/weights/yolov3.weights";
	int inpWidth = 416; // Width of network's input image
        int inpHeight = 416; // Height of networkd's input image

        // Load the network
        Net net = readNetFromDarknet(modelConfiguration, modelWeights);	
net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
        //net.setPreferableBackend(DNN_BACKEND_CUDA);
        //net.setPreferableTarget(DNN_TARGET_OPENCL);
        vector<String> outNames = net.getUnconnectedOutLayersNames();	
	
/*	Mat rgbImageIn;
	cv::Mat rgbImage;
    cv::cvtColor(image,rgbImageIn,cv::COLOR_RGB2BGR);
    resize(rgbImageIn, rgbImage, Size(inpWidth, inpHeight), 1, 1);
   */
string filename = "/home/docker/Projects/DetectionSuite/datasets/coco/oneval2014/COCO_val2014_000000397133.jpg";
        //string filename = "/home/docker/darknet/data/dog.jpg";

        //Mat frame = imread(filename, 0);
        Mat frameIn = imread(filename);
        Mat rgbImage;
        resize(frameIn, rgbImage, Size(inpWidth, inpHeight), 1, 1);

    Mat blob;
    blobFromImage(rgbImage, blob, 1.0, cvSize(rgbImage.cols, rgbImage.rows), Scalar(), true, false, CV_8U);
    net.setInput(blob, "", 0.00392, Scalar());
    // END preprocess
    vector<Mat> outs;
    net.forward(outs, outNames);
    // postprocess

        static vector<int> outLayers = net.getUnconnectedOutLayers();
        static string outLayerType = net.getLayer(outLayers[0])->type;
        vector<int> classIds;
        vector<float> confidences;
	vector<Rect> boxes;
    
    cout << outLayerType << endl;

        for (size_t i = 0; i < outs.size(); i++)
        {
                float* data = (float*)outs[i].data;
                for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * rgbImage.cols);
                    int centerY = (int)(data[1] * rgbImage.rows);
                    int width = (int)(data[2] * rgbImage.cols);
                    int height = (int)(data[3] * rgbImage.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
                cout << "Detection -> " << outs[i] << endl;
        }
        cout << classIds.size() << endl;

        vector<int> indices;
        NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);




	Sample sample;
    RectRegionsPtr regions(new RectRegions());
    ClassTypeGeneric typeConverter(classNamesFile);

	for (size_t i = 0; i < indices.size(); i++) {
                int idx = indices[i];
                Rect box = boxes[idx];
                //rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(0, 255, 0));
                string label = format("%.2f", confidences[idx]);
                typeConverter.setId(idx);
		regions->add(box, typeConverter.getClassString(), confidences[idx]);
		LOG(INFO)<< typeConverter.getClassString() << ": " << confidences[idx] << std::endl;
		
        }



    //DarknetDetections detections = this->cnn->process(rgbImage, (float)confidence_threshold);
/*
    Sample sample;
    RectRegionsPtr regions(new RectRegions());
    ClassTypeGeneric typeConverter(classNamesFile);

    for (auto it = detections.data.begin(), end=detections.data.end(); it !=end; ++it){
        typeConverter.setId(it->classId);
        regions->add(it->detectionBox,typeConverter.getClassString(), it->probability);
        LOG(INFO)<< typeConverter.getClassString() << ": " << it->probability << std::endl;
    }
    sample.setColorImage(image);
    sample.setRectRegions(regions);
    return sample;*/
}
