//
// Created by frivas on 21/01/17.
//

#ifndef SAMPLERGENERATOR_REGION_H
#define SAMPLERGENERATOR_REGION_H


#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>


struct Regions{
    Regions(){};
    // Saves the detections at the path specified in JSON format
    virtual void saveJson(const std::string& outPath)=0;
    virtual void drawRegions(cv::Mat& image)=0;
    virtual void filterSamplesByID(std::vector<std::string> filteredIDS)=0;
    virtual bool empty()=0;
    virtual void print()=0;

};

typedef boost::shared_ptr<Regions> RegionsPtr;

#endif //SAMPLERGENERATOR_REGION_H
