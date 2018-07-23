//
// Created by frivas on 22/01/17.
//

#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <boost/lexical_cast.hpp>
#include "DatasetConverters/readers/OwnDatasetReader.h"

OwnDatasetReader::OwnDatasetReader(const std::string &path,const std::string& classNamesFile,const bool imagesRequired):DatasetReader(imagesRequired){
    this->classNamesFile=classNamesFile;
    appendDataset(path);
}

OwnDatasetReader::OwnDatasetReader(const bool imagesRequired):DatasetReader(imagesRequired) {

}

bool OwnDatasetReader::appendDataset(const std::string &datasetPath, const std::string &datasetPrefix) {
    boost::filesystem::directory_iterator end_itr;
    boost::filesystem::path boostPath(datasetPath);


    std::vector<std::string> filesID;

    for (boost::filesystem::directory_iterator itr(boostPath); itr!=end_itr; ++itr)
    {
        if ((is_regular_file(itr->status()) && itr->path().extension()==".json") && (itr->path().string().find("-region") == std::string::npos)) {
            filesID.push_back(itr->path().filename().stem().string());
        }

    }

    std::sort(filesID.begin(),filesID.end());

    for (auto it = filesID.begin(), end=filesID.end(); it != end; ++it){
        Sample sample(datasetPath,*it);
        sample.setSampleID(datasetPrefix + boost::filesystem::path(*it).filename().stem().string());
        this->samples.push_back(sample);
    }

    LOG(INFO) << "Loaded: " + boost::lexical_cast<std::string>(this->samples.size()) + " samples";
    printDatasetStats();
    return true;
}
