#ifndef SAMPLERGENERATOR_RLEREGIONS_H
#define SAMPLERGENERATOR_RLEREGIONS_H

#include <opencv2/opencv.hpp>
#include "Regions.h"
#include "RleRegion.h"


struct RleRegions:Regions {
    RleRegions();
    void saveJson(const std::string& outPath);
    void add(RLE region, const std::string& classId, const bool isCrowd = false);
    void add(RLE region, const std::string& classId, const double confidence_score, const bool isCrowd = false);
    RleRegion getRegion(int idx);
    std::vector<RleRegion> getRegions();
    void drawRegions(cv::Mat& image);
    void filterSamplesByID(std::vector<std::string> filteredIDS);
    bool empty();
    void print();
    std::vector<RleRegion> regions;
};

typedef boost::shared_ptr<RleRegions> RleRegionsPtr;


#endif //SAMPLERGENERATOR_RLEREGIONS_H
