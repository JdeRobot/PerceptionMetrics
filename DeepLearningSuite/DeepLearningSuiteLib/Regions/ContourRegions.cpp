//
// Created by frivas on 21/01/17.
//

#include "ContourRegions.h"
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <fstream>
#include "rapidjson/filereadstream.h"
#include <glog/logging.h>
ContourRegions::ContourRegions(const std::string &jsonPath) {
    FILE* fp = fopen(jsonPath.c_str(), "rb"); // non-Windows use "r"
    char readBuffer[65536];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
    rapidjson::Document d;
    d.ParseStream(is);
    fclose(fp);
    this->regions.clear();
    for (auto it = d.Begin(), end= d.End(); it != end; ++it){
        std::vector<cv::Point> detection;
        for (auto it2= (*it)["region"].Begin(), end2=(*it)["region"].End(); it2!= end2; ++it2) {
            cv::Point point;
            point.x = (*it2)["x"].GetInt();
            point.y = (*it2)["y"].GetInt();
            detection.push_back(point);
        }
        std::string id = (*it)["id"].GetString();
        this->regions.push_back(ContourRegion(detection,id));
    }
}

ContourRegions::ContourRegions(){

}

void ContourRegions::add(const std::vector<cv::Point> &detections, const std::string& classId, const bool isCrowd) {
    ContourRegion regionToInsert(detections, classId, isCrowd);
    auto itr = std::upper_bound(regions.begin(), regions.end(), regionToInsert);
    regionToInsert.uniqObjectID = regions.size();
    regions.insert(itr, regionToInsert);
}

void ContourRegions::add(const std::vector<cv::Point>& detections, const std::string& classId, const double confidence_score, const bool isCrowd) {
    ContourRegion regionToInsert(detections, classId, confidence_score, isCrowd);
    auto itr = std::upper_bound(regions.begin(), regions.end(), regionToInsert);
    regionToInsert.uniqObjectID = regions.size();
    regions.insert(itr, regionToInsert);
    //regions.push_back(RectRegion(rect, cla
}


void ContourRegions::saveJson(const std::string &outPath) {
    rapidjson::Document d;
    d.SetObject();
    d.SetArray();
    for (auto it = this->regions.begin(), end=this->regions.end(); it != end; it++){
        rapidjson::Value detection;
        detection.SetObject();
        rapidjson::Value idValue(it->classID.c_str(),d.GetAllocator());
        detection.AddMember("classID",idValue,d.GetAllocator());

        rapidjson::Value regionValue;
        regionValue.SetArray();

        for (auto it2=it->region.begin(), end2= it->region.end(); it2 != end2; ++it2) {
            rapidjson::Value point;
            point.SetObject();
            rapidjson::Value xValue(it2->x);
            point.AddMember("x", xValue, d.GetAllocator());

            rapidjson::Value yValue(it2->y);
            point.AddMember("y", yValue, d.GetAllocator());

            regionValue.PushBack(point, d.GetAllocator());
        }
        detection.AddMember("region",regionValue,d.GetAllocator());
        d.PushBack(detection,d.GetAllocator());
    }

    rapidjson::StringBuffer buffer;

    buffer.Clear();

    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    d.Accept(writer);

    std::ofstream outFile(outPath);
    outFile << buffer.GetString() << std::endl;
    outFile.close();
}




ContourRegion ContourRegions::getRegion(int idx) {
    if (this->regions.size() -1 >= idx)
        return this->regions[idx];
    else
        return ContourRegion();
}

void ContourRegions::drawRegions(cv::Mat &image) {
    for (auto it = regions.begin(), end= regions.end(); it != end; ++it) {
        cv::Mat mask = cv::Mat(image.size(), CV_8UC1, cv::Scalar(0));
        cv::Scalar color(255);
        std::vector<std::vector<cv::Point>> contours;
        contours.push_back(it->region);
        cv::drawContours(mask, contours, 0, color, -1, 8);
        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        cv::Mat colorMask(image.size(), CV_8UC1, cv::Scalar(255));
        colorMask.copyTo(channels[0], mask);
        colorMask.copyTo(channels[1], mask);
        cv::Mat image2show;
        cv::merge(channels, image2show);
        image2show.copyTo(image);

    }

}

std::vector<ContourRegion> ContourRegions::getRegions() {
    return this->regions;
}

void ContourRegions::filterSamplesByID(std::vector<std::string> filteredIDS) {
    std::vector<ContourRegion> oldRegions(this->regions);
    this->regions.clear();
    for(auto it = oldRegions.begin(), end=oldRegions.end(); it != end; ++it) {
        if (std::find(filteredIDS.begin(), filteredIDS.end(), it->classID) != filteredIDS.end()) {
            this->regions.push_back(*it);
        }
    }
}

bool ContourRegions::empty() {
    return (this->regions.size()==0);
}

void ContourRegions::print() {
    //todo
    LOG(ERROR) << "Not yet implemented" << std::endl;
}
