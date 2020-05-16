#ifndef SAMPLERGENERATOR_SAMPLESREADER_H
#define SAMPLERGENERATOR_SAMPLESREADER_H


#include <DatasetConverters/readers/DatasetReader.h>

class SamplesReader: public DatasetReader {
public:
    SamplesReader(std::vector<Sample> & samples, std::string &classNamesFile);

    //bool appendDataset(const std::string& datasetPath, const std::string& datasetPrefix="");

};

typedef boost::shared_ptr<SamplesReader> SamplesReaderPtr;

#endif //SAMPLERGENERATOR_SAMPLESREADER_H
