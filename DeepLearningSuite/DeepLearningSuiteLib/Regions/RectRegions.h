//
// Created by frivas on 21/01/17.
//

#ifndef SAMPLERGENERATOR_RECTREGIONS_H
#define SAMPLERGENERATOR_RECTREGIONS_H

#include <opencv2/opencv.hpp>
#include "Regions.h"
#include "RectRegion.h"
#include <algorithm>

struct RectRegions:Regions {
    RectRegions(const std::string& jsonPath);
    RectRegions();

    void add(const cv::Rect_<double> rect, const std::string classId, const bool isCrowd = false);
    void add(const cv::Rect_<double> rect, const std::string classId, const double confidence_score, const bool isCrowd = false);
    void add(const std::vector<cv::Point_<double>>& detections, const std::string classId, const bool isCrowd = false);
    void add(double x, double y, double w, double h, const std::string classId, const bool isCrowd = false);
    void add(double x, double y, double w, double h, const std::string classId, const double confidence_score, const bool isCrowd = false);

    void saveJson(const std::string& outPath);
    RectRegion getRegion(int id);
    std::vector<RectRegion> getRegions();
    void drawRegions(cv::Mat& image);
    void filterSamplesByID(std::vector<std::string> filteredIDS);
    bool empty();
    void print();

    std::vector<RectRegion> regions;
};

typedef boost::shared_ptr<RectRegions> RectRegionsPtr;


#endif //SAMPLERGENERATOR_RECTREGIONS_H
