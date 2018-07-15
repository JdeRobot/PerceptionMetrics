#ifndef SAMPLERGENERATOR_COCODATASETREADER_H
#define SAMPLERGENERATOR_COCODATASETREADER_H


#include <DatasetConverters/readers/DatasetReader.h>
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include "rapidjson/error/en.h"
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

class COCODatasetReader: public DatasetReader {
public:
    COCODatasetReader(const std::string& path,const std::string& classNamesFile);
    COCODatasetReader();
    bool appendDataset(const std::string& datasetPath, const std::string& datasetPrefix="");
    bool find_img_directory(const boost::filesystem::path & dir_path, boost::filesystem::path & path_found, std::string& img_file_prefix,  int& longestSubSeq);
private:
    std::map<unsigned long int,unsigned long int> map_image_id;

};

typedef boost::shared_ptr<COCODatasetReader> COCODatasetReaderPtr;

#endif //SAMPLERGENERATOR_COCODATASETREADER_H
