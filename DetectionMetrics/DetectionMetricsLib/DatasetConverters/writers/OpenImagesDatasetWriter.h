#ifndef SAMPLERGENERATOR_OPENIMAGESDATASETCONVERTER_H
#define SAMPLERGENERATOR_OPENIMAGESDATASETCONVERTER_H

#include <string>
#include "DatasetWriter.h"
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

class OpenImagesDatasetWriter: public DatasetWriter {
public:
    OpenImagesDatasetWriter(const std::string& outPath, DatasetReaderPtr& reader, const std::string& writerNamesFile, bool overWriteclassWithZero=false);
    void process(bool writeImages = false, bool useDepth = false);

private:
    std::string fullImagesPath;
    std::string fullLabelsPath;
    std::string fullNamesPath;
    bool overWriteclassWithZero;
    std::string writerNamesFile;
};

typedef  boost::shared_ptr<OpenImagesDatasetWriter> OpenImagesDatasetWriterPtr;


#endif
