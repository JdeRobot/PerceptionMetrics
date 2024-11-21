//
// Created by frivas on 28/01/17.
//

#ifndef SAMPLERGENERATOR_DETECTOR_H
#define SAMPLERGENERATOR_DETECTOR_H

#include <SampleGeneratorLib/Sample.h>

class Detector{
public:
    Detector(){};

    virtual Sample inferImage(const cv::Mat& image)=0;
};

#endif //SAMPLERGENERATOR_DETECTOR_H
