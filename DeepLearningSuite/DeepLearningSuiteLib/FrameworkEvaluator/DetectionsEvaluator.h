//
// Created by frivas on 1/02/17.
//

#ifndef SAMPLERGENERATOR_DETECTIONSEVALUATOR_H
#define SAMPLERGENERATOR_DETECTIONSEVALUATOR_H

#include <DatasetConverters/readers/DatasetReader.h>
#include <boost/shared_ptr.hpp>
#include "ClassStatistics.h"
#include "GlobalStats.h"
#include <DatasetConverters/ClassTypeMapper.h>
#include <Common/EvalMatrix.h>
#include <unordered_map>

class DetectionsEvaluator {
public:
    DetectionsEvaluator(DatasetReaderPtr gt, DatasetReaderPtr detections, bool debug=true);
    void evaluate();
    void addValidMixClass(const std::string classA, const std::string classB);
    void addClassToDisplay(const std::string& classID);
    GlobalStats getStats();

private:
    DatasetReaderPtr gt;
    DatasetReaderPtr detections;
    bool debug;
    std::vector<std::pair<std::string, std::string>> validMixClass;
    std::unordered_map<std::string, std::string> classMapping;

    void evaluateSamples(Sample gt, Sample detection);
    void evaluateSample(Sample gt, Sample detection, Eval::EvalMatrix& evalmatrix);

    void printStats();

    bool sameClass(const std::string class1, const std::string class2);

    std::vector<std::string> classesToDisplay;
    double thIOU;
    GlobalStats stats;

    double iouThrs[10] = {0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95};
};


typedef boost::shared_ptr<DetectionsEvaluator> DetectionsEvaluatorPtr;


#endif //SAMPLERGENERATOR_DETECTIONSEVALUATOR_H
