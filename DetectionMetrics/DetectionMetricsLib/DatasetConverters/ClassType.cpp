//
// Created by frivas on 2/02/17.
//

#include "ClassType.h"

float colorsClass[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };


cv::Scalar ClassType::getColor() {

    auto itFind = std::find(this->classes.begin(), this->classes.end(),this->classID);
    int id = std::distance(this->classes.begin(),itFind);
    int nClasses= this->classes.size();

    int offset = id*123457 % nClasses;
    float red = _get_color(2,offset,nClasses);
    float green = _get_color(1,offset,nClasses);
    float blue = _get_color(0,offset,nClasses);
    return cv::Scalar(red*255, green*255,blue*255);
}

float ClassType::_get_color(int c, int x, int max) {

    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colorsClass[i][c] + ratio*colorsClass[j][c];
    //printf("%f\n", r);
    return r;
}

//ClassType::ClassType(int id) {
//    fillStringClassesVector();
//    this->classID=this->classes[id];
//}
//
//ClassType::ClassType(const std::string &classID) {
//    fillStringClassesVector();
//    this->classID=classID;
//}

std::string ClassType::getClassString() {
    return this->classID;
}

int ClassType::getClassID() {
    auto itFind = std::find(this->classes.begin(), this->classes.end(),this->classID);
    int id = std::distance(this->classes.begin(),itFind);
    return id;
}

std::vector<std::string> ClassType::getAllAvailableClasses() {
    return this->classes;
}
