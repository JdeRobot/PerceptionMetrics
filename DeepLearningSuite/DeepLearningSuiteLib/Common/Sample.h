//
// Created by frivas on 22/01/17.
//

#ifndef SAMPLERGENERATOR_SAMPLE_H
#define SAMPLERGENERATOR_SAMPLE_H

#include <Regions/RectRegions.h>
#include <Regions/ContourRegions.h>
#include <Regions/RleRegions.h>

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
    void setSampleDims(const int width, const int height);
    void setColorImage(const std::string& imagePath);
    void setColorImage(const cv::Mat& image);
    void setDepthImage(const cv::Mat& image);
    void setDepthImage(const std::string& imagePath);
    void setRectRegions(const RectRegionsPtr& regions);
    void setContourRegions(const ContourRegionsPtr& regions);
    void setRleRegions(const RleRegionsPtr& regions);
    void setSampleID(const std::string& sampleID);
    void clearColorImage();                         // For better memory management
    void clearDepthImage();                         // For better memeory management

    int getSampleWidth()const;
    int getSampleHeight()const;
    RectRegionsPtr getRectRegions()const;
    ContourRegionsPtr getContourRegions()const;
    RleRegionsPtr getRleRegions()const;
    std::string getColorImagePath() const;
    std::string getDepthImagePath() const;
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
    void print();
    bool show(const std::string readerImplementation, const std::string windowName, const int waitKey, const bool showDepth);

    bool isDepthImageValid();
    bool isValid();
    void filterSamplesByID(std::vector<std::string> filteredIDS);
    std::string getSampleID();



private:
    RectRegionsPtr rectRegions;
    ContourRegionsPtr contourRegions;
    RleRegionsPtr rleRegions;
    cv::Mat colorImage;
    std::string colorImagePath;
    cv::Mat depthImage;
    std::string depthImagePath;
    std::string sampleID;
    int width = -1;
    int height = -1;
};


#endif //SAMPLERGENERATOR_SAMPLE_H
