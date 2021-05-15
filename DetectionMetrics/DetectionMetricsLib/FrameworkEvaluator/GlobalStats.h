//
// Created by frivas on 22/07/17.
//

#ifndef SAMPLERGENERATOR_GLOBALSTATS_H
#define SAMPLERGENERATOR_GLOBALSTATS_H

#include <map>
#include "ClassStatistics.h"

class GlobalStats {
public:
    GlobalStats();
    void addTruePositive(const std::string &classID, double confScore);

    void addFalsePositive(const std::string &classID, double confScore);

    void addFalseNegative(const std::string &classID);

    void addGroundTruth(const std::string &classID, bool isRegular);

    void addIgnore(const std::string &classID, double confScore);

    void addIOU(const std::string &classID, double value);

    //void printStats(const std::vector<std::string>& classesToDisplay) const;

    std::map<std::string,ClassStatistics> getStats() const;

private:
    std::map<std::string,ClassStatistics> statsMap;

};


#endif //SAMPLERGENERATOR_GLOBALSTATS_H
