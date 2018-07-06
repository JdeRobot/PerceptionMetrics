//
// Created by frivas on 22/07/17.
//

#include "GlobalStats.h"
#include <iostream>

GlobalStats::GlobalStats() = default;

void GlobalStats::addIgnore(const std::string &classID, double confScore ) {
    //std::cout << "Ignoring " << classID << "keihfiewoaiasssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss" << '\n';
    if (this->statsMap.count(classID)){
        auto it = this->statsMap[classID].confScores.insert(-confScore);
        unsigned int index = std::distance(this->statsMap[classID].confScores.begin(), it);
        auto itr = this->statsMap[classID].truePositives.begin();
        this->statsMap[classID].truePositives.insert(itr + index, 0);
        itr = this->statsMap[classID].falsePositives.begin();
        this->statsMap[classID].falsePositives.insert(itr + index, 0);
    }
    else{
        ClassStatistics s(classID);
        s.truePositives.push_back(0);
        s.falsePositives.push_back(0);
        s.confScores.insert(-confScore);
        this->statsMap[classID]=s;
    }
}

void GlobalStats::addGroundTruth(const std::string &classID, bool isRegular) {
    if (this->statsMap.count(classID)) {
        if (isRegular)
            this->statsMap[classID].numGroundTruthsReg++;
        else
            this->statsMap[classID].numGroundTruthsIg++;
    } else {
        ClassStatistics s(classID);
        if (isRegular)
            s.numGroundTruthsReg++;
        else
            s.numGroundTruthsIg++;
        this->statsMap[classID] = s;
    }
}

void GlobalStats::addTruePositive(const std::string &classID, double confScore) {
    if (this->statsMap.count(classID)){
        auto it = this->statsMap[classID].confScores.insert(-confScore);
        unsigned int index = std::distance(this->statsMap[classID].confScores.begin(), it);
        auto itr = this->statsMap[classID].truePositives.begin();
        this->statsMap[classID].truePositives.insert(itr + index, 1);
        itr = this->statsMap[classID].falsePositives.begin();
        this->statsMap[classID].falsePositives.insert(itr + index, 0);
    }
    else{
        ClassStatistics s(classID);
        s.truePositives.push_back(1);
        s.falsePositives.push_back(0);
        s.confScores.insert(-confScore);
        this->statsMap[classID]=s;
    }
}

void GlobalStats::addFalsePositive(const std::string &classID, double confScore) {
    //std::cout << "Adding False positive: " << classID << " " << confScore <<'\n';
    if (this->statsMap.count(classID)){

        auto it = this->statsMap[classID].confScores.insert(-confScore);
        unsigned int index = std::distance(this->statsMap[classID].confScores.begin(), it);
        auto itr = this->statsMap[classID].truePositives.begin();
        this->statsMap[classID].truePositives.insert(itr + index, 0);
        itr = this->statsMap[classID].falsePositives.begin();
        this->statsMap[classID].falsePositives.insert(itr + index, 1);
    }
    else{
        ClassStatistics s(classID);
        s.truePositives.push_back(0);
        s.falsePositives.push_back(1);
        s.confScores.insert(-confScore);
        this->statsMap[classID]=s;
    }
}

void GlobalStats::addFalseNegative(const std::string &classID)  {
    if (this->statsMap.count(classID)){
        this->statsMap[classID].falseNegatives = this->statsMap[classID].falseNegatives+1;
    }
    else{
        ClassStatistics s(classID);
        s.falseNegatives = s.falseNegatives+1;
        this->statsMap[classID]=s;
    }
}

void GlobalStats::addIOU(const std::string &classID, double value)  {
    if (this->statsMap.count(classID)){
        this->statsMap[classID].iou.push_back(value);
    }
    else{
        ClassStatistics s(classID);
        s.iou.push_back(value);
        this->statsMap[classID]=s;
    }
}

/*void GlobalStats::printStats(const std::vector<std::string>& classesToDisplay) const{
    if (classesToDisplay.empty()) {
        for (auto it : this->statsMap) {
            it.second.printStats();
        }
    }
    else{
        for (const std::string& it : classesToDisplay){
            if (this->statsMap.count(it)) {
                auto data =this->statsMap.at(it);
                data.printStats();
            }
        }
    }
}*/

std::map<std::string,ClassStatistics> GlobalStats::getStats() const{
    return statsMap;
}
