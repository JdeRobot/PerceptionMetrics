//
// Created by frivas on 30/01/17.
//

#ifndef SAMPLERGENERATOR_STRINGHANDLER_H
#define SAMPLERGENERATOR_STRINGHANDLER_H


#include <string>
#include <sstream>
#include <vector>


class StringHandler {
private:
    template<typename Out>
    static void split(const std::string &s, char delim, Out result) {
        std::stringstream ss;
        ss.str(s);
        std::string item;
        while (std::getline(ss, item, delim)) {
            *(result++) = item;
        }
    }

public:
    static std::vector<std::string> split(const std::string &s, char delim) {
        std::vector<std::string> elems;
        split(s, delim, std::back_inserter(elems));
        return elems;
    }
};


#endif //SAMPLERGENERATOR_STRINGHANDLER_H
