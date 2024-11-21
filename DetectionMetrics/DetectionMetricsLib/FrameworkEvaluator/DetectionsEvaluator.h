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
#include <tuple>
#include <valarray>

class DetectionsEvaluator {
public:
    DetectionsEvaluator(DatasetReaderPtr gt, DatasetReaderPtr detections, bool debug=false);
    void evaluate(bool isIouTypeBbox);
    void accumulateResults();
    void addValidMixClass(const std::string classA, const std::string classB);
    void addClassToDisplay(const std::string& classID);
    std::map<std::string, double> getClassWiseAP();
    std::map<std::string, double> getClassWiseAR();
    double getOverallmAP();
    double getOverallmAR();
    double getEvaluationTime();
    double getAccumulationTime();

private:
    DatasetReaderPtr gt;
    DatasetReaderPtr detections;
    bool debug;
    std::vector<std::pair<std::string, std::string>> validMixClass;
    std::unordered_map<std::string, std::string> classMapping;

    //void evaluateSamples(Sample gt, Sample detection);
    void evaluateSample(Sample gt, Sample detection, bool isIouTypeBbox);

    void printStats();

    bool sameClass(const std::string class1, const std::string class2);

    std::vector<std::string> classesToDisplay;
    double thIOU;
    std::map<double, GlobalStats> sampleStats;

    GlobalStats stats;

    std::map<std::string, double> classWiseMeanAP;
    std::map<std::string, double> classWiseMeanAR;
    std::valarray<double> ApDiffIou = std::valarray<double>(10);
    std::valarray<double> ArDiffIou = std::valarray<double>(10);


    std::map<std::string, std::tuple <unsigned int, unsigned int>> areaRng = { {"all", std::make_tuple(0, 10000000000) },
                                                                {"small", std::make_tuple(0, 1024) },
                                                                {"medium", std::make_tuple(1024, 9216) },
                                                                {"large", std::make_tuple(9210, 10000000000)} };

    double iouThrs[10] = {0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95};

    std::vector<double> recallThrs;         // 101 recall Thrs initialized in constructor
    
    
    double timeEvaluation = 0;
    double timeAccumulation = 0;
    
};


typedef boost::shared_ptr<DetectionsEvaluator> DetectionsEvaluatorPtr;


#endif //SAMPLERGENERATOR_DETECTIONSEVALUATOR_H
