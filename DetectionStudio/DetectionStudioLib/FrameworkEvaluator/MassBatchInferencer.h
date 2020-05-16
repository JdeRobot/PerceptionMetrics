#ifndef SAMPLERGENERATOR_MASSBATCHINFERENCER_H
#define SAMPLERGENERATOR_MASSBATCHINFERENCER_H

#include <DatasetConverters/readers/DatasetReader.h>
#include <FrameworkEvaluator/FrameworkInferencer.h>

class MassBatchInferencer {
public:
    MassBatchInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, const std::string& resultsPath, double confidence_threshold = 0.2, bool debug=true);
    MassBatchInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, double confidence_threshold = 0.2, bool debug=true);
    void process(const int batchSize, bool writeImages, DatasetReaderPtr readerDetection = NULL);

private:
    DatasetReaderPtr reader;
    FrameworkInferencerPtr inferencer;
    std::string resultsPath;
    bool debug;
    bool saveOutput;
    int alreadyProcessed;
    double confidence_threshold;

};


#endif //SAMPLERGENERATOR_MASSINFERENCER_H
