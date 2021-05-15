//
// Created by frivas on 5/02/17.
//

#ifndef SAMPLERGENERATOR_DATASETWRITTER_H
#define SAMPLERGENERATOR_DATASETWRITTER_H


#include <DatasetConverters/readers/DatasetReader.h>
#include <unordered_map>

class DatasetWriter {
public:
    DatasetWriter(const std::string& outPath, DatasetReaderPtr& reader);
    virtual void process(bool writeImages = false, bool useDepth = false)=0;

protected:
    std::string outPath;
    DatasetReaderPtr& reader;
    std::vector<std::string> outputClasses;
    std::unordered_map<std::string, std::string> mapped_classes;
    std::unordered_map<std::string, long int> discarded_classes;
    unsigned int skip_count = 10;  //max Number of annotations that can be skipped if Corresponding images weren't found
};


typedef  boost::shared_ptr<DatasetWriter> DatasetWriterPtr;

#endif //SAMPLERGENERATOR_DATASETWRITTER_H
