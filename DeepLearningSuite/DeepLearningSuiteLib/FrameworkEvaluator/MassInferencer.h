//
// Created by frivas on 1/02/17.
//

#ifndef SAMPLERGENERATOR_MASSINFERENCER_H
#define SAMPLERGENERATOR_MASSINFERENCER_H

#include <DatasetConverters/readers/DatasetReader.h>
#include <FrameworkEvaluator/FrameworkInferencer.h>

class MassInferencer {
public:
    MassInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, const std::string& resultsPath, double* confidence_threshold = NULL, bool debug=true);
    MassInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, const std::string& resultsPath, bool debug=true);
    MassInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, const std::string &resultsPath, bool* stopDeployer, double* confidence_threshold = NULL, bool debug=true);
    MassInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, double* confidence_threshold = NULL, bool debug=true);
    MassInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, bool debug=true);
    void process(bool useDepthImages, std::vector<Sample>* samples = NULL);

private:
    DatasetReaderPtr reader;
    FrameworkInferencerPtr inferencer;
    std::string resultsPath;
    bool debug;
    bool saveOutput;
    int alreadyProcessed;
    bool* stopDeployer = NULL;
    double* confidence_threshold = NULL;
    double default_confidence_threshold = 0.2;

};



#endif //SAMPLERGENERATOR_MASSINFERENCER_H
