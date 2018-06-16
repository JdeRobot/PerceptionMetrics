#ifndef SAMPLERGENERATOR_CAFFEINFERENCER_H
#define SAMPLERGENERATOR_CAFFEINFERENCER_H


#include "FrameworkInferencer.h"
#include <vector>
#include <fstream>
#include <sstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class CaffeInferencer: public FrameworkInferencer {
public:
    CaffeInferencer(const std::string& netConfig, const std::string& netWeights, const std::string& classNamesFile);
    Sample detectImp(const cv::Mat& image);
    std::vector<cv::String> getOutputsNames();
    void postprocess(const std::vector<cv::Mat>& outs);

private:
    std::string netConfig;
    std::string netWeights;
    struct detection {
        cv::Rect boundingBox;
        float probability;
        int classId;
    };

    std::vector<detection> detections;
    double confThreshold;
    cv::dnn::Net net;
};


typedef boost::shared_ptr<CaffeInferencer> CaffeInferencerPtr;

#endif //SAMPLERGENERATOR_CAFFEINFERENCER_H
