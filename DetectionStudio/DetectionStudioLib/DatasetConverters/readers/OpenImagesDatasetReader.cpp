#include <fstream>
#include <glog/logging.h>
#include <boost/filesystem/path.hpp>
#include "OpenImagesDatasetReader.h"
#include "DatasetConverters/ClassTypeGeneric.h"

using namespace boost::filesystem;



OpenImagesDatasetReader::OpenImagesDatasetReader(const std::string &path,const std::string& classNamesFile, bool imagesRequired):DatasetReader(imagesRequired) {
    this->classNamesFile=classNamesFile;
    appendDataset(path);
}

bool OpenImagesDatasetReader::find_img_directory( const path & dir_path, path & path_found ) {
    LOG(INFO) << dir_path.string() << '\n';
	
}

bool OpenImagesDatasetReader::appendDataset(const std::string &datasetPath, const std::string &datasetPrefix) {
    LOG(INFO) << "Dataset Path: " << datasetPath << '\n';
}

