//
// Created by frivas on 20/01/17.
//

#ifndef SAMPLERGENERATOR_DETECTIONSVALIDATOR_H
#define SAMPLERGENERATOR_DETECTIONSVALIDATOR_H


#include <opencv2/opencv.hpp>
#include <Common/Sample.h>

class DetectionsValidator {
public:
    DetectionsValidator(const std::string& pathToSave, double scale=3);
    ~DetectionsValidator();
    void validate(const cv::Mat& colorImage,const cv::Mat& depthImage, std::vector<std::vector<cv::Point>>& detections);
    void validate(const Sample& inputSample);

private:
    int validationCounter;
    std::string path;
    double scale;

    void fillRectIntoImageDimensions(cv::Rect_<double>& rect, const cv::Size size);
};


#endif //SAMPLERGENERATOR_DETECTIONSVALIDATOR_H
