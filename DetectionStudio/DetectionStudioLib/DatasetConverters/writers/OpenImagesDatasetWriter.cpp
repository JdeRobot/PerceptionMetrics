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
    // Write output to .csv file
    //
    // Add images to folder

    Sample sample;
    std::string labelFilePath= this->fullLabelsPath + "/" + "instances_labels.csv";
    std::ofstream out(labelFilePath);
    out << "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside" << std::endl;

    while (reader->getNextSample(sample)){
        auto boundingBoxes = sample.getRectRegions()->getRegions();
	std::string sampleId = sample.getSampleID();
	
	//LOG(INFO) << "IMAGE PATH: " << sample.getColorImagePath() << "\n";

	for (auto it = boundingBoxes.begin(), end=boundingBoxes.end(); it != end; ++it){
	    double x = it->region.x;
            double y = it->region.y;
            double w = it->region.width;
            double h = it->region.height;
            double confidence_score = it->confidence_score;
	    std::string classId = it->classID;
	    LOG(INFO) << "x: " << x << " y: " << y << " w: " << w << " h: " << h << " confidence: " << confidence_score << " class id: " << it->classID << "\n";
	    out << sampleId << "," << "xclick" << "," << classId << "," << confidence_score << "," << x << "," << y << "," << w << "," << h << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << std::endl;
	}

    }
    if (!out.good()) throw std::runtime_error ("Can't write to the file!");
}
