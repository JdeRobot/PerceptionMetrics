#ifndef SAMPLERGENERATOR_RLEREGION_H
#define SAMPLERGENERATOR_RLEREGION_H

#include <boost/shared_ptr.hpp>
#include "maskApi.h"

struct RleRegion {
    RleRegion():valid(false){};
    RleRegion(const RLE region, std::string classID,
        bool isCrowd = false):region(region),classID(classID),isCrowd(isCrowd),valid(true){}; //person by default
    RleRegion(const RLE region, std::string classID,
        double confidence_score, bool isCrowd = false):region(region),classID(classID),confidence_score(confidence_score),isCrowd(isCrowd),valid(true){};


    bool operator < (const RleRegion &obj) const {

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

     RLE region;

     std::string classID;
     bool isCrowd = false;      // Can be substantial for COCO dataset, which ignores iscrowd in evaluations
     long double area;          // This can be either Bounding Box area or Contour Area, necessary for
                                // determining area Range in evaluations, and may be directly read from
                                // dataset like COCO.
     int uniqObjectID;
     double confidence_score = -1;
     bool valid;


};


#endif //SAMPLERGENERATOR_RLEREGION_H
