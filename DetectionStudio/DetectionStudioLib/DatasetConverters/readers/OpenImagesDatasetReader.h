#ifndef SAMPLERGENERATOR_OPENIMAGESDATASETREADER_H
#define SAMPLERGENERATOR_OPENIMAGESDATASETREADER_H


#include <DatasetConverters/readers/DatasetReader.h>
#include <boost/filesystem/path.hpp>


class OpenImagesDatasetReader: public DatasetReader {
	public:
		OpenImagesDatasetReader(const std::string &path,const std::string& classNamesFile, bool imagesRequired);
		bool appendDataset(const std::string& datasetPath, const std::string& datasetPrefix="");
		bool find_img_directory( const boost::filesystem::path & ann_dir_path, boost::filesystem::path & path_found);


};


typedef boost::shared_ptr<OpenImagesDatasetReader> OpenImagesDatasetReaderPtr;



#endif //SAMPLERGENERATOR_OPENIMAGESDATASETREADER_H
