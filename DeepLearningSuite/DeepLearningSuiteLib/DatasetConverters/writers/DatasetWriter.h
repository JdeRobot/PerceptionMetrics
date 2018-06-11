//
// Created by frivas on 5/02/17.
//

#ifndef SAMPLERGENERATOR_DATASETWRITTER_H
#define SAMPLERGENERATOR_DATASETWRITTER_H


#include <DatasetConverters/readers/DatasetReader.h>

class DatasetWriter {
public:
    DatasetWriter(const std::string& outPath, DatasetReaderPtr& reader);
    virtual void process(bool usedColorImage=true)=0;

protected:
    std::string outPath;
    DatasetReaderPtr& reader;
    std::vector<std::string> outputClasses;
    std::unordered_map<std::string, std::string> mapped_classes;
    std::unordered_map<std::string, long int> discarded_classes;
};


typedef  boost::shared_ptr<DatasetWriter> DatasetWriterPtr;

#endif //SAMPLERGENERATOR_DATASETWRITTER_H
