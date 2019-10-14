//
// Created by frivas on 9/02/17.
//

#ifndef SAMPLERGENERATOR_CLASSTYPEGENERIC_H
#define SAMPLERGENERATOR_CLASSTYPEGENERIC_H


#include "ClassType.h"

struct ClassTypeGeneric: public ClassType{
    ClassTypeGeneric(const std::string& classesFile);
    ClassTypeGeneric(const std::string& classesFile, int id);
    void setId(int id);
    void fillStringClassesVector(const std::string& classesFile);
    void Print();
};


#endif //SAMPLERGENERATOR_CLASSTYPEGENERIC_H
