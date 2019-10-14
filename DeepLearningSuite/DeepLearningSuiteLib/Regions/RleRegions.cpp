#include "RleRegions.h"
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <fstream>
#include <random>
#include "rapidjson/filereadstream.h"
#include <glog/logging.h>

RleRegions::RleRegions(){

}

void RleRegions::add(RLE region, const std::string& classId, const bool isCrowd) {
    RleRegion regionToInsert(region, classId, isCrowd);
    auto itr = std::upper_bound(regions.begin(), regions.end(), regionToInsert);
    regionToInsert.uniqObjectID = regions.size();
    regions.insert(itr, regionToInsert);
}

void RleRegions::add(RLE region, const std::string& classId, const double confidence_score, const bool isCrowd) {
    RleRegion regionToInsert(region, classId, confidence_score, isCrowd);
    auto itr = std::upper_bound(regions.begin(), regions.end(), regionToInsert);
    regionToInsert.uniqObjectID = regions.size();
    regions.insert(itr, regionToInsert);
    //regions.push_back(RectRegion(rect, cla
}


void RleRegions::saveJson(const std::string &outPath) {
    rapidjson::Document d;
}




RleRegion RleRegions::getRegion(int idx) {
    if (this->regions.size() -1 >= idx)
        return this->regions[idx];
    else
        return RleRegion();
}

void RleRegions::drawRegions(cv::Mat &image) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    for (auto it = regions.begin(), end= regions.end(); it != end; ++it) {

        cv::Mat mask = cv::Mat(it->region.w, it->region.h, CV_8UC1, cv::Scalar(0));

        rleDecode(&(it->region), mask.data , 1);
        mask = mask * 255;
        cv::Mat rotatedMask = mask.t();

        cv::Scalar color;
        std::vector<std::vector<cv::Point> > contours;
        if (it->isCrowd) {
            color = cv::Scalar(2,166,101);
        } else {
            color = cv::Scalar((unsigned int)(distribution(generator)*170), (unsigned int)(distribution(generator)*170), (unsigned int)(distribution(generator)*170));
            cv::findContours( rotatedMask.clone(), contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

        }

        cv::Mat colorMask(image.size(), CV_8UC3, color);

        cv::Mat output(colorMask.size(), CV_8UC3, cv::Scalar(0));
        colorMask.copyTo(output, rotatedMask);

        image = image.mul((( 255 - output )/255 ))  + output;
        cv::drawContours(image, contours, -1, color, 2, 8);

    }

}

std::vector<RleRegion> RleRegions::getRegions() {
    return this->regions;
}

void RleRegions::filterSamplesByID(std::vector<std::string> filteredIDS) {
    std::vector<RleRegion> oldRegions(this->regions);
    this->regions.clear();
    for(auto it = oldRegions.begin(), end=oldRegions.end(); it != end; ++it) {
        if (std::find(filteredIDS.begin(), filteredIDS.end(), it->classID) != filteredIDS.end()) {
            this->regions.push_back(*it);
        }
    }
}

bool RleRegions::empty() {
    return (this->regions.size()==0);
}

void RleRegions::print() {
    //todo
    LOG(ERROR) << "Not yet implemented" << std::endl;
}
