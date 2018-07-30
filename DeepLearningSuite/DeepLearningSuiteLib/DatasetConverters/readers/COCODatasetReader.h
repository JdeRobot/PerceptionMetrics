#ifndef SAMPLERGENERATOR_COCODATASETREADER_H
#define SAMPLERGENERATOR_COCODATASETREADER_H


#include <DatasetConverters/readers/DatasetReader.h>
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include "rapidjson/error/en.h"
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include "DatasetConverters/ClassTypeGeneric.h"

class COCODatasetReader: public DatasetReader {
public:
    COCODatasetReader(const std::string& path,const std::string& classNamesFile, bool imagesRequired);
    bool appendDataset(const std::string& datasetPath, const std::string& datasetPrefix="");
    bool find_img_directory(const boost::filesystem::path & dir_path, boost::filesystem::path & path_found, std::string& img_filename);

    void appendSegmentationRegion(const rapidjson::Value& node, Sample& sample, ClassTypeGeneric typeConverter, const bool isCrowd);

    RLE fromSegmentationList(const rapidjson::Value& seg, int im_width, int im_height, int size = 1);
    RLE getSegmentationRegion(const rapidjson::Value& seg, int im_width, int im_height);
    RLE fromSegmentationObject(const rapidjson::Value& seg, int size = 1);
    RLE fromUncompressedRle(const rapidjson::Value& seg);
    RLE fromRle(const rapidjson::Value& seg);
private:
    std::map < unsigned long int, Sample > map_image_id;      // map image id to sample, helps storage in a sorted way

};

typedef boost::shared_ptr<COCODatasetReader> COCODatasetReaderPtr;

#endif //SAMPLERGENERATOR_COCODATASETREADER_H
