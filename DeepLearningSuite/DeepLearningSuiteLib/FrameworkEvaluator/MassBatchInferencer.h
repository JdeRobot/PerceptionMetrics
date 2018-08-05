#ifndef SAMPLERGENERATOR_MASSINFERENCER_H
#define SAMPLERGENERATOR_MASSINFERENCER_H

#include <DatasetConverters/readers/DatasetReader.h>
#include <FrameworkEvaluator/FrameworkInferencer.h>

class MassInferencer {
public:
    MassBatchInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, const std::string& resultsPath, double* confidence_threshold = NULL, bool debug=true);
    MassBatchInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, const std::string& resultsPath, bool debug=true);
    MassBatchInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, const std::string &resultsPath, bool* stopDeployer, double* confidence_threshold = NULL, bool debug=true);
    MassBatchInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, double* confidence_threshold = NULL, bool debug=true);
    MassBatchInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, bool debug=true);
    void process(int batch_size, bool writeImages, DatasetReaderPtr readerDetection = NULL);

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
