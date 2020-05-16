//
// Created by frivas on 30/01/17.
//

#include "Normalizations.h"




void Normalizations::normalizeRect(cv::Rect &region, cv::Size size) {
    normalizeLower(region.x);
    normalizeLower(region.y);
    normalizeUpper(region.x, region.width, size.width);
    normalizeUpper(region.y, region.height,size.height);

}

void Normalizations::normalizeLower(int &value, int min) {
    if (value < min){
        value=min;
    }
}

void Normalizations::normalizeUpper(int pos, int& size, int max) {
    if (pos + size >= max){
        size= (max - pos - 1);
    }
}
