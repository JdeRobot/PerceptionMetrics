//
// Created by frivas on 21/01/17.
//

#ifndef SAMPLERGENERATOR_CONTOURREGIONS_H
#define SAMPLERGENERATOR_CONTOURREGIONS_H

#include <opencv2/opencv.hpp>
#include "Regions.h"
#include "ContourRegion.h"


struct ContourRegions:Regions {
    ContourRegions(const std::string& jsonPath);
    ContourRegions();
    void saveJson(const std::string& outPath);
    void add(const std::vector<cv::Point>& detections, const std::string& classId, const bool isCrowd = false);
    void add(const std::vector<cv::Point>& detections, const std::string& classId, const double confidence_score, const bool isCrowd = false);
    ContourRegion getRegion(int idx);
    std::vector<ContourRegion> getRegions();
    void drawRegions(cv::Mat& image);
    void filterSamplesByID(std::vector<std::string> filteredIDS);
    bool empty();
    void print();
    std::vector<ContourRegion> regions;
};

typedef boost::shared_ptr<ContourRegions> ContourRegionsPtr;


#endif //SAMPLERGENERATOR_CONTOURREGIONS_H
