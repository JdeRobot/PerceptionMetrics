//
// Created by frivas on 2/02/17.
//

#ifndef SAMPLERGENERATOR_CLASSTYPEOWN_H
#define SAMPLERGENERATOR_CLASSTYPEOWN_H


#include "ClassType.h"

struct ClassTypeOwn: public ClassType{
    ClassTypeOwn(int id);
    ClassTypeOwn(const std::string& classID);
    void fillStringClassesVector();

};


#endif //SAMPLERGENERATOR_CLASSTYPEOWN_H
