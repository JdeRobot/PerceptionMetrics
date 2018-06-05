#ifndef SAMPLERGENERATOR_COCODATASETREADER_H
#define SAMPLERGENERATOR_COCODATASETREADER_H


#include <DatasetConverters/readers/DatasetReader.h>
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>


class COCODatasetReader: public DatasetReader {
public:
    COCODatasetReader(const std::string& path,const std::string& classNamesFile);
    COCODatasetReader();
    bool appendDataset(const std::string& datasetPath, const std::string& datasetPrefix="");
private:
    std::map<unsigned long int,unsigned long int> map_image_id;

};

typedef boost::shared_ptr<COCODatasetReader> COCODatasetReaderPtr;

#endif //SAMPLERGENERATOR_COCODATASETREADER_H
