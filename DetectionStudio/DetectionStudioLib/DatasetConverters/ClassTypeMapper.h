#ifndef SAMPLERGENERATOR_CLASSTYPEMAPPER_H
#define SAMPLERGENERATOR_CLASSTYPEMAPPER_H


#include "ClassType.h"
#include "Tree.h"
#include <algorithm>
#include <unordered_map>

struct ClassTypeMapper: public ClassType{
    ClassTypeMapper(const std::string& classNamesFile);
    ClassTypeMapper();
    void fillStringClassesVector(const std::string &classesFile);
    bool mapString(std::string &className);
    std::unordered_map<std::string, std::string> mapFile(std::string classNamesFile);

private:
    Tree root;
};


#endif //SAMPLERGENERATOR_CLASSTYPEMAPPER_H
