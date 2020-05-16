#ifndef SAMPLERGENERATOR_PASCALVOCDATASETCONVERTER_H
#define SAMPLERGENERATOR_PASCALVOCDATASETCONVERTER_H

#include <string>
#include "DatasetWriter.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/foreach.hpp>

class PascalVOCDatasetWriter: public DatasetWriter {
public:
    PascalVOCDatasetWriter(const std::string& outPath, DatasetReaderPtr& reader, const std::string& writerNamesFile, bool overWriteclassWithZero=false);
    void process(bool writeImages = false, bool useDepth = false);

private:
    std::string fullImagesPath;
    std::string fullLabelsPath;
    std::string fullNamesPath;
    bool overWriteclassWithZero;
    std::string writerNamesFile;
};

typedef  boost::shared_ptr<PascalVOCDatasetWriter> PascalVOCDatasetWriterPtr;


#endif //SAMPLERGENERATOR_PASCALVOCDATASETCONVERTER_H
