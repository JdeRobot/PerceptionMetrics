#ifndef SAMPLERGENERATOR_CLASSTYPEMAPPER_H
#define SAMPLERGENERATOR_CLASSTYPEMAPPER_H


#include "ClassType.h"
#include "Tree.h"
#include <algorithm>

struct ClassTypeMapper: public ClassType{
    ClassTypeMapper(const std::string& classNamesFile);
    ClassTypeMapper();
    void fillStringClassesVector(const std::string &classesFile);
    bool mapString(std::string className);

private:
    Tree root;
};


#endif //SAMPLERGENERATOR_CLASSTYPEMAPPER_H
