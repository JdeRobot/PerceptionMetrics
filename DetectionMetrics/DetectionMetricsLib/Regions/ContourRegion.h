//
// Created by frivas on 26/01/17.
//

#ifndef SAMPLERGENERATOR_CONTOURREGION_H
#define SAMPLERGENERATOR_CONTOURREGION_H

#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>

struct ContourRegion {
    ContourRegion():valid(false){};
    ContourRegion(const ContourRegion& other);
    ContourRegion(const std::vector<cv::Point>& region, std::string classID,
        bool isCrowd = false):region(region),classID(classID),isCrowd(isCrowd),valid(true){}; //person by default
    ContourRegion(const std::vector<cv::Point>& region, std::string classID,
        double confidence_score, bool isCrowd = false):region(region),classID(classID),confidence_score(confidence_score),isCrowd(isCrowd),valid(true){};


    bool operator < (const ContourRegion &obj) const {

        if (classID.empty() || obj.classID.empty()) {
           throw std::invalid_argument("One of the ContourRegions passed for comparision were not initialized");
        }

        if (classID != obj.classID) {
           //std::cout << "returning not equal class" << '\n';
           return classID < obj.classID;

        } else {
           //std::cout << "came here" << '\n';
           if (isCrowd || obj.isCrowd) {
              return (isCrowd ^ obj.isCrowd) & (!isCrowd);
           }
           return confidence_score > obj.confidence_score;          //Reverse Sorting of Confidence Scores
        }

     }

     std::vector<cv::Point>region;

     std::string classID;
     bool isCrowd = false;      // Can be substantial for COCO dataset, which ignores iscrowd in evaluations
     long double area;          // This can be either Bounding Box area or Contour Area, necessary for
                                // determining area Range in evaluations, and may be directly read from
                                // dataset like COCO.
     int uniqObjectID;
     double confidence_score = -1;
     bool valid;


};


#endif //SAMPLERGENERATOR_CONTOURREGION_H
