//
// Created by frivas on 25/01/17.
//

#ifndef SAMPLERGENERATOR_RECTREGION_H
#define SAMPLERGENERATOR_RECTREGION_H

#include <opencv2/opencv.hpp>

struct RectRegion {

    RectRegion():valid(false){};
    RectRegion(const cv::Rect_<double>& region, const std::string& classID, const bool isCrowd = false):region(region),classID(classID),valid(true),isCrowd(isCrowd){};
    RectRegion(const cv::Rect_<double>& region, const std::string& classID, const double confidence_score, const bool isCrowd = false):region(region),classID(classID),confidence_score(confidence_score),valid(true),isCrowd(isCrowd){};

    bool operator < (const RectRegion &obj) const {

      if (classID.empty() || obj.classID.empty()) {
         throw std::invalid_argument("One of the RectRegions passed for comparision were not initialized, ClassID found empty");
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

    cv::Rect_<double> region;
    cv::Rect_<double> nameRect;
    std::string classID;
    bool isCrowd = false;      // Can be substantial for COCO dataset, which ignores iscrowd in evaluations
    long double area;          // This can be either Bounding Box area or Contour Area, necessary for
                               // determining area Range in evaluations, and may be directly read from
                               // dataset like COCO.
    int uniqObjectID;
    double confidence_score = -1;
    bool valid;

};


#endif //SAMPLERGENERATOR_RECTREGION_H
