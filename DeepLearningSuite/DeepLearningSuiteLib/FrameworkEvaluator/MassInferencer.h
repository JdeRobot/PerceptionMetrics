//
// Created by frivas on 1/02/17.
//

#ifndef SAMPLERGENERATOR_MASSINFERENCER_H
#define SAMPLERGENERATOR_MASSINFERENCER_H

#include <DatasetConverters/readers/DatasetReader.h>
#include <FrameworkEvaluator/FrameworkInferencer.h>

class MassInferencer {
public:
    MassInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, const std::string& resultsPath, bool debug=true);
    void process(bool useDepthImages, std::vector<Sample>* samples = NULL);

private:
    DatasetReaderPtr reader;
    FrameworkInferencerPtr inferencer;
    std::string resultsPath;
    bool debug;
    int alreadyProcessed;

};



#endif //SAMPLERGENERATOR_MASSINFERENCER_H
