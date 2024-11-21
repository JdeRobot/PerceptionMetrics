//
// Created by frivas on 19/07/17.
//

#ifndef SAMPLERGENERATOR_PATHHELPER_H
#define SAMPLERGENERATOR_PATHHELPER_H

#include <string>


class PathHelper {
    public:
        static std::string concatPaths(const std::string& p1, const std::string& p2);
        static std::string getPathSeparator();

};


#endif //SAMPLERGENERATOR_PATHHELPER_H
