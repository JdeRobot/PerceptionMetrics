//
// Created by frivas on 19/07/17.
//

#include <boost/algorithm/string/predicate.hpp>
#include "PathHelper.h"

std::string PathHelper::concatPaths(const std::string &p1, const std::string &p2) {
    if (boost::algorithm::ends_with(p1, getPathSeparator())){
        return p1 + p2;
    }
    else{
        return p1 + getPathSeparator() + p2;
    }
}

std::string PathHelper::getPathSeparator() {
#ifdef __linux__
    return std::string("/");
#else
    return std::string("\\");
#endif
}
