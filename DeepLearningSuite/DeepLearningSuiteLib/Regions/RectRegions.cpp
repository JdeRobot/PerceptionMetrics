//
// Created by frivas on 21/01/17.
//

#include "RectRegions.h"
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <fstream>
#include "rapidjson/filereadstream.h"
#include <DatasetConverters/ClassTypeOwn.h>
#include <glog/logging.h>


RectRegions::RectRegions(const std::string &jsonPath) {
    FILE* fp = fopen(jsonPath.c_str(), "rb"); // non-Windows use "r"
    char readBuffer[65536];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
    rapidjson::Document d;
    d.ParseStream(is);
    fclose(fp);
    for (auto it=d.Begin(), end = d.End(); it != end; ++it){
        cv::Rect reg((*it)["x"].GetInt(),(*it)["y"].GetInt(),(*it)["w"].GetInt(),(*it)["h"].GetInt());
        std::string id;
        id= (*it)["id"].GetString();
        regions.push_back(RectRegion(reg,id));
    }
}
RectRegions::RectRegions() {

}


void RectRegions::saveJson(const std::string &outPath) {
    rapidjson::Document d;
    d.SetArray();
    for (auto it = regions.begin(), end= regions.end(); it != end; ++it){
        rapidjson::Value node;
        node.SetObject();
        rapidjson::Value xValue(it->region.x);
        node.AddMember("x",xValue,d.GetAllocator());

        rapidjson::Value yValue(it->region.y);
        node.AddMember("y",yValue,d.GetAllocator());

        rapidjson::Value wValue(it->region.width);
        node.AddMember("w",wValue,d.GetAllocator());

        rapidjson::Value hValue(it->region.height);
        node.AddMember("h",hValue,d.GetAllocator());

        rapidjson::Value confValue(it->confidence_score);
        node.AddMember("score",confValue,d.GetAllocator());

        rapidjson::Value idValue(it->classID.c_str(), d.GetAllocator());
        node.AddMember("id",idValue,d.GetAllocator());

        d.PushBack(node,d.GetAllocator());
    }



    rapidjson::StringBuffer buffer;

    buffer.Clear();

    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    d.Accept(writer);

    std::ofstream outFile(outPath);
    outFile << buffer.GetString() << std::endl;
    outFile.close();
}

void RectRegions::add(const cv::Rect_<double> rect,const std::string classId, bool isCrowd){
    RectRegion regionToInsert(rect, classId, isCrowd);
    auto itr = std::upper_bound(regions.begin(), regions.end(), regionToInsert);
    regionToInsert.uniqObjectID = regions.size();
    regions.insert(itr, regionToInsert);

    //regions.push_back(RectRegion(rect,classId));
}

void RectRegions::add(const cv::Rect_<double> rect, const std::string classId, double confidence_score, bool isCrowd) {
    RectRegion regionToInsert(rect, classId, confidence_score, isCrowd);
    auto itr = std::upper_bound(regions.begin(), regions.end(), regionToInsert);
    regionToInsert.uniqObjectID = regions.size();
    regions.insert(itr, regionToInsert);
    //regions.push_back(RectRegion(rect, classId, confidence_score));
}


void RectRegions::add(const std::vector<cv::Point_<double>> &detections,const std::string classId, const bool isCrowd) {
    regions.push_back(RectRegion(cv::boundingRect(detections),classId, isCrowd));
}

void RectRegions::add(double x, double y, double w, double h,const std::string classId, const bool isCrowd) {
    RectRegion regionToInsert(cv::Rect_<double>(x,y,w,h), classId, isCrowd);
    auto itr = std::upper_bound(regions.begin(), regions.end(), regionToInsert);
    regionToInsert.uniqObjectID = regions.size();
    regions.insert(itr, regionToInsert);
    //regions.push_back(RectRegion(cv::Rect(x,y,w,h),classId));
}

void RectRegions::add(double x, double y, double w, double h, const std::string classId, const double confidence_score, const bool isCrowd) {
    RectRegion regionToInsert(cv::Rect_<double>(x,y,w,h), classId, confidence_score, isCrowd);
    auto itr = std::upper_bound(regions.begin(), regions.end(), regionToInsert);
    regionToInsert.uniqObjectID = regions.size();
    regions.insert(itr, regionToInsert);
}

RectRegion RectRegions::getRegion(int id) {
    if (this->regions.size() -1 >= id)
        return this->regions[id];
    else
        return RectRegion();
}

void RectRegions::drawRegions(cv::Mat &image) {
    if (!image.empty())
        for (auto it = regions.begin(), end=regions.end(); it != end; ++it) {
            ClassTypeOwn classType(it->classID);
            cv::rectangle(image, it->region, classType.getColor(), 2);
            cv::Size rectSize(80,20);
            cv::Rect nameRectangle(it->region.x, it->region.y - rectSize.height, rectSize.width,rectSize.height);
            if (nameRectangle.y < 0){
                nameRectangle.y=it->region.y;
            }
            if (nameRectangle.x + nameRectangle.width > image.size().width){
                nameRectangle.x = image.size().width - nameRectangle.width -1;
            }
            if (nameRectangle.y + nameRectangle.height > image.size().height){
                nameRectangle.y = image.size().height - nameRectangle.height -1;
            }

            if (nameRectangle.x<0)
                nameRectangle.x=0;
            if (nameRectangle.y<0)
                nameRectangle.y=0;

            image(nameRectangle)=cv::Scalar(classType.getColor());
            cv::putText(image, classType.getClassString(),cv::Point(nameRectangle.x - nameRectangle.height/4 + 5 ,nameRectangle.y + nameRectangle.height - 5),cv::FONT_HERSHEY_TRIPLEX,0.4,cv::Scalar(0,0,0),1);
        }

}

std::vector<RectRegion> RectRegions::getRegions() {
    return this->regions;
}

void RectRegions::filterSamplesByID(std::vector<std::string> filteredIDS) {
    std::vector<RectRegion> oldRegions(this->regions);
    this->regions.clear();
    for(auto it = oldRegions.begin(), end=oldRegions.end(); it != end; ++it) {
        if (std::find(filteredIDS.begin(), filteredIDS.end(), it->classID) != filteredIDS.end()) {
            this->regions.push_back(*it);
        }
    }
}

bool RectRegions::empty() {
    return (this->regions.size()==0);
}

void RectRegions::print() {
    LOG(INFO) << "-------------------" << std::endl;
    for (auto it = this->regions.begin(), end = this->regions.end(); it != end; ++it){
        int idx = std::distance(this->regions.begin(),it);
        LOG(INFO) << "[" << idx << "]: " << it->region << " (" << it->classID << ")" << std::endl;
    }
    LOG(INFO) << "-------------------" << std::endl;
}
