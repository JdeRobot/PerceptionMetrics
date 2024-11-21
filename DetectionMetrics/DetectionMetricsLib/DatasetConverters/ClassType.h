//
// Created by frivas on 2/02/17.
//

#ifndef SAMPLERGENERATOR_CLASSTYPE_H
#define SAMPLERGENERATOR_CLASSTYPE_H

#include <opencv2/opencv.hpp>

struct ClassType {

    cv::Scalar getColor();
    std::string getClassString();
    int getClassID();
    std::vector<std::string> getAllAvailableClasses();
protected:
    float _get_color(int c, int x, int max);
    std::vector<std::string> classes;
    std::string classID;


};


#endif //SAMPLERGENERATOR_CLASSTYPE_H
