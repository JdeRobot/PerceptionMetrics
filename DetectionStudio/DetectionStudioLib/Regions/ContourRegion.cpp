//
// Created by frivas on 26/01/17.
//

#include "ContourRegion.h"

ContourRegion::ContourRegion(const ContourRegion &other) {
    this->classID=other.classID;
    if (other.region.size()) {
        this->region.resize(other.region.size());
        std::copy(other.region.begin(), other.region.end(), this->region.begin());
    }
    this->valid=other.valid;
}
