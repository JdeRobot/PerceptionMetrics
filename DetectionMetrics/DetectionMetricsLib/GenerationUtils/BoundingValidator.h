//
// Created by frivas on 22/11/16.
//

#ifndef SAMPLERGENERATOR_BOUNDINGVALIDATOR_H
#define SAMPLERGENERATOR_BOUNDINGVALIDATOR_H


#include <opencv2/opencv.hpp>
#include <Regions/RectRegion.h>
#include "BoundingRectGuiMover.h"

static bool clicked;
static cv::Point from, to;
static cv::Point tempFrom;
static BoundingRectGuiMover::MovementType movementType;

class BoundingValidator {
public:
    explicit BoundingValidator(const cv::Mat& image_in, double scale=3);
    bool validate(std::vector<cv::Point>& bounding,cv::Rect_<double>& validatedBound, int& key);
    bool validate(const cv::Rect_<double>& bounding,cv::Rect_<double>& validatedBound, int& key);
    bool validateNDetections(std::vector<RectRegion>& regions);


private:
    double scale;
    cv::Mat image;


    static void CallBackFunc(int event, int x, int y, int flags, void* userdata);
    static void CallBackFuncNumberDetections(int event, int x, int y, int flags, void* userdata);





    cv::Mat updateRegionsImage(const std::vector<RectRegion>& regions);


};


#endif //SAMPLERGENERATOR_BOUNDINGVALIDATOR_H
