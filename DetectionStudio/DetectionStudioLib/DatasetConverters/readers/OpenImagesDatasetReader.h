#ifndef SAMPLERGENERATOR_OPENIMAGESDATASETREADER_H
#define SAMPLERGENERATOR_OPENIMAGESDATASETREADER_H


#include <DatasetConverters/readers/DatasetReader.h>
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include "rapidjson/error/en.h"
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include "DatasetConverters/ClassTypeGeneric.h"

class OpenImagesDatasetReader: public DatasetReader {
	public:
		OpenImagesDatasetReader(const std::string &path,const std::string& classNamesFile, bool imagesRequired);
		bool appendDataset(const std::string& datasetPath, const std::string& datasetPrefix="");
		bool find_img_directory( const boost::filesystem::path & ann_dir_path, boost::filesystem::path & path_found);
	private:
    		std::map < unsigned long int, Sample > map_image_id;      // map image id to sample, helps storage in a sorted way
};


typedef boost::shared_ptr<OpenImagesDatasetReader> OpenImagesDatasetReaderPtr;



#endif //SAMPLERGENERATOR_OPENIMAGESDATASETREADER_H
