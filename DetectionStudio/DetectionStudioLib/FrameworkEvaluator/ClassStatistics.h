//
// Created by frivas on 1/02/17.
//

#ifndef SAMPLERGENERATOR_STATISTICS_H
#define SAMPLERGENERATOR_STATISTICS_H


#include <vector>
#include <string>
#include <set>
#include <numeric>
#include <functional>
#include <algorithm>
#include <type_traits>

struct ClassStatistics {
    ClassStatistics();
    ClassStatistics(const std::string& classID);
    double getMeanIOU() const;
    double getAveragePrecision(std::vector<double> recallThrs) const;
    std::vector<double> getPrecisionArray() const;
    std::vector<double> getPrecisionArrayOp() const;
    std::vector<double> getPrecisionForDiffRecallThrs(std::vector<double> recallThrs) const;
    std::vector<double> getRecallArray() const;
    double getRecall() const;
    static double divide(double x, double y);
    //void printStats() const;


    std::string classID;
    int numGroundTruthsReg = 0;
    int numGroundTruthsIg = 0;
    std::vector<double> iou;
    std::vector<int> truePositives;
    std::vector<int> falsePositives;
    std::multiset<double> confScores;
    // 2 vectors in same order on with conf score and other with true positive 1/0
    int nSamples;
    //int truePositives;
    //int falsePositives;
    int falseNegatives;
    int trueNegatives; //???? evaluar muestra negativa??


};


#endif //SAMPLERGENERATOR_STATISTICS_H
