//
// Created by frivas on 22/01/17.
//

#ifndef SAMPLERGENERATOR_SAMPLE_H
#define SAMPLERGENERATOR_SAMPLE_H

#include <Regions/RectRegions.h>
#include <Regions/ContourRegions.h>

struct Sample {
    Sample(const std::string& path, const std::string& id, bool loadDepth=true);
    Sample();
    Sample(const cv::Mat& colorImage);
    Sample(const cv::Mat& colorImage, const RectRegionsPtr& rectRegions);
    Sample(const cv::Mat& colorImage, const ContourRegionsPtr& contourRegions);
    Sample(const cv::Mat& colorImage, const RectRegionsPtr& rectRegions, const ContourRegionsPtr& contourRegions);
    Sample(const cv::Mat& colorImage, const cv::Mat& depthImage, const RectRegionsPtr& rectRegions);
    Sample(const cv::Mat& colorImage, const cv::Mat& depthImage, const ContourRegionsPtr& contourRegions);
    Sample(const cv::Mat& colorImage, const cv::Mat& depthImage, const RectRegionsPtr& rectRegions, const ContourRegionsPtr& contourRegions);
    ~Sample();
    void setColorImage(const std::string& imagePath);
    void setColorImage(const cv::Mat& image);
    void setDepthImage(const cv::Mat& image);
    void setDepthImage(const std::string& imagePath);
    void setRectRegions(const RectRegionsPtr& regions);
    void setContourRegions(const ContourRegionsPtr& regions);
    void setSampleID(const std::string& sampleID);

    RectRegionsPtr getRectRegions()const;
    ContourRegionsPtr getContourRegions();
    cv::Mat getColorImage() const;
    cv::Mat getDepthImage() const;
    cv::Mat getDeptImageGrayRGB() const;
    cv::Mat getSampledColorImage() const;
    cv::Mat getSampledDepthImage() const;
    cv::Mat getSampledDepthColorMapImage(double alpha = 1 , double beta = 0) const;
    cv::Mat getDepthColorMapImage(double alpha = 1 , double beta = 0) const;
    void save(const std::string& outPath, int id);
    void save(const std::string& outPath, const std::string& filename);
    void save(const std::string& outPath);

    bool isDepthImageValid();
    bool isValid();
    void filterSamplesByID(std::vector<std::string> filteredIDS);
    std::string getSampleID();



private:
    RectRegionsPtr rectRegions;
    ContourRegionsPtr contourRegions;
    cv::Mat colorImage;
    std::string colorImagePath;
    cv::Mat depthImage;
    std::string depthImagePath;
    std::string sampleID;
};


#endif //SAMPLERGENERATOR_SAMPLE_H
