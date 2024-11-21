//
// Created by frivas on 17/11/16.
//

#ifndef SAMPLERGENERATOR_DEPTHFOREGROUNDSEGMENTADOR_H
#define SAMPLERGENERATOR_DEPTHFOREGROUNDSEGMENTADOR_H
#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>
#include <DepthFilter.h>
#include <opencv2/core/version.hpp>

class DepthForegroundSegmentator {
public:
    DepthForegroundSegmentator(bool filterActive=true);
    std::vector<std::vector<cv::Point>>process(const cv::Mat& image);
    cv::Mat process2(const cv::Mat& image);



private:
#if  CV_MAJOR_VERSION == 3
    cv::BackgroundSubtractorMOG2* bg;
#else
    cv::Ptr<cv::BackgroundSubtractor> bg;
#endif
    cv::Mat fore;
    bool filterActive;
    boost::shared_ptr<jderobot::DepthFilter> filter;
    double defaultLearningRate;
    double minBlobArea;
};


#endif //SAMPLERGENERATOR_DEPTHFOREGROUNDSEGMENTADOR_H
