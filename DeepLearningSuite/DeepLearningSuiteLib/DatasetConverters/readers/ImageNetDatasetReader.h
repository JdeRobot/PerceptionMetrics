#ifndef SAMPLERGENERATOR_IMAGENETDATASETREADER_H
#define SAMPLERGENERATOR_IMAGENETDATASETREADER_H


#include <DatasetConverters/readers/DatasetReader.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/foreach.hpp>

class ImageNetDatasetReader: public DatasetReader {
public:
    ImageNetDatasetReader(const std::string& path,const std::string& classNamesFile, bool imagesRequired);
    ImageNetDatasetReader(const std::string& classNamesFile, bool imagesRequired);
    bool appendDataset(const std::string& datasetPath, const std::string& datasetPrefix="");
    bool find_img_directory( const boost::filesystem::path & ann_dir_path, boost::filesystem::path & path_found );
    bool find_directory(const boost::filesystem::path & dir_path, const std::string & dir_name, boost::filesystem::path & path_found);

};

typedef boost::shared_ptr<ImageNetDatasetReader> ImageNetDatasetReaderPtr;

#endif //SAMPLERGENERATOR_IMAGENETDATASETREADER_H
