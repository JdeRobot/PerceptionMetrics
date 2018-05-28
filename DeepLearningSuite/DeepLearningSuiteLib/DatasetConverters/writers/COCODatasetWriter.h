#ifndef SAMPLERGENERATOR_COCODATASETCONVERTER_H
#define SAMPLERGENERATOR_COCODATASETCONVERTER_H

#include <string>
#include "DatasetWriter.h"
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

class COCODatasetWriter: public DatasetWriter {
public:
    COCODatasetWriter(const std::string& outPath, DatasetReaderPtr& reader, bool overWriteclassWithZero=true);
    void process(bool usedColorImage=true);

private:
    std::string fullImagesPath;
    std::string fullLabelsPath;
    bool overWriteclassWithZero;
};

typedef  boost::shared_ptr<COCODatasetWriter> COCODatasetWriterPtr;


#endif
