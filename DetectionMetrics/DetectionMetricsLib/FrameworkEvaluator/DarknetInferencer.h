//
// Created by frivas on 31/01/17.
//

#ifndef SAMPLERGENERATOR_DARKNETEVALUATOR_H
#define SAMPLERGENERATOR_DARKNETEVALUATOR_H

#include <boost/shared_ptr.hpp>
#include "FrameworkInferencer.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

class DarknetInferencer: public FrameworkInferencer {
public:
    DarknetInferencer(const string& netConfig, const string& netWeights, const string& classNamesFile);
    Sample detectImp(const Mat& image, double confidence_threshold);

private:
    string netConfig;
    string netWeights;
    vector<string> classes;
    Net net;
    vector<String> outNames;
    float nmsThreshold;
};

typedef boost::shared_ptr<DarknetInferencer> DarknetInferencerPtr;

#endif //SAMPLERGENERATOR_DARKNETEVALUATOR_H
