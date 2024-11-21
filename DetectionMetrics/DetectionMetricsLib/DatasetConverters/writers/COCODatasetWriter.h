#ifndef SAMPLERGENERATOR_COCODATASETCONVERTER_H
#define SAMPLERGENERATOR_COCODATASETCONVERTER_H

#include <string>
#include "DatasetWriter.h"
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

class COCODatasetWriter: public DatasetWriter {
public:
    COCODatasetWriter(const std::string& outPath, DatasetReaderPtr& reader, const std::string& writerNamesFile, bool overWriteclassWithZero=false);
    void process(bool writeImages = false, bool useDepth = false);

private:
    std::string fullImagesPath;
    std::string fullLabelsPath;
    std::string fullNamesPath;
    bool overWriteclassWithZero;
    std::string writerNamesFile;
};

typedef  boost::shared_ptr<COCODatasetWriter> COCODatasetWriterPtr;


#endif
