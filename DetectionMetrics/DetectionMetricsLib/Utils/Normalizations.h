//
// Created by frivas on 30/01/17.
//

#ifndef SAMPLERGENERATOR_NORMALIZATIONS_H
#define SAMPLERGENERATOR_NORMALIZATIONS_H

#include <opencv2/opencv.hpp>

class Normalizations {
public:
    static void normalizeRect(cv::Rect& region, cv::Size size);

private:
    static void normalizeLower(int& value, int min=0);
    static void normalizeUpper(int pos, int& size, int max);
};


#endif //SAMPLERGENERATOR_NORMALIZATIONS_H
