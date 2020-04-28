//
// Created by frivas on 31/01/17.
//
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <chrono>
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
 	this->netConfig=netConfig;
	this->netWeights=netWeights;

	std::vector<string> classes = {};
        ifstream ifs(this->classNamesFile.c_str());
	string line;
        while (getline(ifs, line)) classes.push_back(line);
	this->classes=classes;

    	// Load the network
	Net net = readNetFromDarknet(this->netConfig, this->netWeights);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
	//net.setPreferableTarget(DNN_TARGET_OPENCL);
	this->net=net;
	this->outNames=net.getUnconnectedOutLayersNames();
	this->nmsThreshold = 0.4;
}

Sample DarknetInferencer::detectImp(const cv::Mat &image, double confThreshold) {
	int inpWidth = (image.cols/32) * 32; // Width of network's input image
	int inpHeight = (image.rows/32) * 32;;

        Mat rgbImage;
        resize(image, rgbImage, Size(inpWidth, inpHeight), 1, 1);

    	Mat blob;
    	blobFromImage(rgbImage, blob, 1.0, cvSize(inpWidth, inpHeight), Scalar(), true, false, CV_8U);
    	//blobFromImage(image, blob, 1.0, cvSize(image.cols, image.rows), Scalar(), true, false, CV_8U);
	net.setInput(blob, "", 0.00392, Scalar());
    	
	// END preprocess
    	vector<Mat> outs;
	cout << "Starting inference" << endl;
	auto start = std::chrono::system_clock::now();
    	net.forward(outs, outNames);
	auto end = std::chrono::system_clock::now();
	chrono::duration<double> elapsed_seconds = end-start;
        cout << "Inference Time: " << elapsed_seconds.count() << " seconds" << endl;
    	// postprocess

        vector<int> classIds;
        vector<float> confidences;
	vector<Rect> boxes;
    
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
				int centerX = (int)(data[0] * image.cols);
                                int centerY = (int)(data[1] * image.rows);
                                int width = (int)(data[2] * image.cols);
                                int height = (int)(data[3] * image.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

                    		classIds.push_back(classIdPoint.x);
                    		confidences.push_back((float)confidence);
                    		boxes.push_back(Rect(left, top, width, height));
                	}
            	}
        }
	cout << "Num Detections: " << classIds.size() << endl;
        vector<int> indices;
        NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

	Sample sample;
    	RectRegionsPtr regions(new RectRegions());
	for (size_t i = 0; i < indices.size(); i++) {
                int idx = indices[i];
                Rect box = boxes[idx];
                string label = this->classes[classIds[idx]];
		regions->add(box, label, confidences[idx]);
        	LOG(INFO)<< label << ": " << confidences[idx] << std::endl;
	}

	sample.setColorImage(image);
	sample.setRectRegions(regions);
	return sample;
}
