//
// Created by frivas on 20/02/17.
//

#include <FrameworkEvaluator/DetectionsEvaluator.h>
#include "Evaluator.h"
#include "SamplerGenerationHandler.h"

void
SampleGeneratorHandler::Evaluator::process(QListView *datasetListGT, QListView *namesListGT, QListView *readerImpListGT,
                                           QListView *datasetListDetect, QListView *namesListDetect,
                                           QListView *readerImpListDetect, QListView *filterClasses,
                                           const std::string &datasetPath, const std::string &namesGTPath,
                                           const std::string &inferencesPath, const std::string &inferencesNamesPath,bool overWriterPersonClasses,bool enableMixEvaluation,bool showEval) {

    GenericDatasetReaderPtr readerGT = SamplerGenerationHandler::createDatasetReaderPtr(datasetListGT, namesListGT,
                                                                                        readerImpListGT, filterClasses,
                                                                                        datasetPath,
                                                                                        inferencesNamesPath);
    GenericDatasetReaderPtr readerDetection = SamplerGenerationHandler::createDatasetReaderPtr(datasetListDetect,
                                                                                               namesListDetect,
                                                                                               readerImpListDetect,
                                                                                               filterClasses,
                                                                                               inferencesPath,
                                                                                               inferencesNamesPath);



    DetectionsEvaluatorPtr evaluator(new DetectionsEvaluator(readerGT->getReader(),readerDetection->getReader(),showEval));


    if (overWriterPersonClasses){
        readerGT->getReader()->overWriteClasses("person-falling","person");
        readerGT->getReader()->overWriteClasses("person-fall","person");
        readerGT->getReader()->printDatasetStats();
    }

    if(enableMixEvaluation) {
        evaluator->addValidMixClass("person", "person-falling");
        evaluator->addValidMixClass("person", "person-fall");
    }
    evaluator->evaluate();
    evaluator->accumulateResults();

}
