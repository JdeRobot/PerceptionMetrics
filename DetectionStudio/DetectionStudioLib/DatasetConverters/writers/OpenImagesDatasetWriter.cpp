#include "OpenImagesDatasetWriter.h"
#include "DatasetConverters/ClassTypeMapper.h"
#include <iomanip>
#include <math.h>
#include <fstream>
#include <algorithm>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <glog/logging.h>

using namespace rapidjson;

OpenImagesDatasetWriter::OpenImagesDatasetWriter(const std::string &outPath, DatasetReaderPtr &reader, const std::string& writerNamesFile, bool overWriteclassWithZero):DatasetWriter(outPath,reader),overWriteclassWithZero(overWriteclassWithZero), writerNamesFile(writerNamesFile){
    
    this->fullImagesPath=boost::filesystem::absolute(boost::filesystem::path(outPath + "/images")).string();
    this->fullLabelsPath=boost::filesystem::absolute(boost::filesystem::path(outPath + "/labels")).string();


    auto boostImages= boost::filesystem::path(fullImagesPath);
    if (!boost::filesystem::exists(boostImages)){
        boost::filesystem::create_directories(boostImages);
    }
    auto boostLabels= boost::filesystem::path(fullLabelsPath);
    if (!boost::filesystem::exists(boostLabels)){
        boost::filesystem::create_directories(boostLabels);
    }

    LOG(INFO) << "Full images path: " << this->fullImagesPath << std::endl;
    LOG(INFO) << "Full labels path: " << this->fullLabelsPath << std::endl;
}



void OpenImagesDatasetWriter::process(bool writeImages, bool useDepth) {
    LOG(INFO) << "--2--" << "\n";

    // Write output to .csv file
    //
    // Add images to folder
    //
    //
    //

    Sample sample;

    while (reader->getNextSample(sample)){
        auto boundingBoxes = sample.getRectRegions()->getRegions();
	//LOG(INFO) << "IMAGE PATH: " << sample.getColorImagePath() << "\n";

	for (auto it = boundingBoxes.begin(), end=boundingBoxes.end(); it != end; ++it){
	     double x = it->region.x;
            double y = it->region.y;
            double w = it->region.width;
            double h = it->region.height;
            double confidence_score = it->confidence_score;
	    LOG(INFO) << "x: " << x << " y: " << y << " w: " << w << " h: " << h << " confidence: " << confidence_score << " class id: " << it->classID << "\n";
	}


    }



}
