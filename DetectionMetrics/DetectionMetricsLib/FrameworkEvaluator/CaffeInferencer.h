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
    CaffeInferencer(const std::string& netConfig, const std::string& netWeights, const std::string& classNamesFile, std::map<std::string, std::string>* inferencerParamsMap);
    Sample detectImp(const cv::Mat& image, double confidence_threshold);
    std::vector<cv::String> getOutputsNames();
    void postprocess(const std::vector<cv::Mat>& outs, cv::Mat& image, double confidence_threshold);


private:
    std::string netConfig;
    std::string netWeights;
    struct detection {
        cv::Rect boundingBox;
        float probability;
        int classId;
    };

    std::vector<detection> detections;
    std::vector<cv::String> names;
    double scaling_factor;
    cv::Scalar mean_sub;
    cv::dnn::Net net;
    int inpWidth;
    int inpHeight;
    bool swapRB;

};


typedef boost::shared_ptr<CaffeInferencer> CaffeInferencerPtr;

#endif //SAMPLERGENERATOR_CAFFEINFERENCER_H
