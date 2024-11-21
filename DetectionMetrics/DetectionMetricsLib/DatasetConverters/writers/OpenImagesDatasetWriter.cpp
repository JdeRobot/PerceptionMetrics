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
    Sample sample;
    ClassTypeMapper typeMapper;

    if (!writerNamesFile.empty())
        typeMapper = ClassTypeMapper(writerNamesFile);


    std::string labelFilePath= this->fullLabelsPath + "/" + "instances_labels.csv";
    std::ofstream out(labelFilePath);
    out << "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside" << std::endl;

    while (reader->getNextSample(sample)){
        auto boundingBoxes = sample.getRectRegions()->getRegions();
	std::string sampleId = sample.getSampleID();
	std::string imageFilePath= this->fullImagesPath + "/" + sampleId + ".jpg";
       
        // Write images in case of converting dataset	
	cv::Mat image;
        if (writeImages) {
            if (useDepth) {
                image = sample.getDepthImage();
            } else {
                image = sample.getColorImage();
            }
            if (image.empty()) {
                skip_count++;
                if (skip_count > this->skip_count) {
                    throw std::runtime_error("Maximum limit for skipping exceeded, either turn off writing images or fix issues in dataset");
                }
                LOG(WARNING) << "Image empty, skipping writing image. Skipped " + std::to_string(skip_count) + " of " + std::to_string(this->skip_count);
            } else {
                cv::imwrite(imageFilePath,image);
            }
        }	

	for (auto it = boundingBoxes.begin(), end=boundingBoxes.end(); it != end; ++it){
            std::string classId = it->classID;
	    if (writerNamesFile.empty()) {
		std::vector<std::string>::iterator itr;
                itr = find(this->outputClasses.begin(), this->outputClasses.end(), it->classID);
                if (itr == this->outputClasses.end()) {
                    this->outputClasses.push_back(it->classID);
                    classId = this->outputClasses.size() - 1;
                } else {
                    classId = std::distance(this->outputClasses.begin(), itr);
                }
	    } else {
		// Try mapping class name if the network classes are different from input dataset
		if(typeMapper.mapString(it->classID)) {         // Mapping Successfull
		    classId = typeMapper.getClassString().substr(0, typeMapper.getClassString().find(","));
                } else {                                        // No Mapping Found Discarding Class
	            LOG(INFO) << "no MAPPING" << "\n";		
                    std::unordered_map<std::string, long int>::iterator itr = this->discarded_classes.find(it->classID);
                    if (itr != this->discarded_classes.end()) {
                        itr->second++;
                    } else {
                        this->discarded_classes.insert(std::make_pair(it->classID, 1));
		    }
                    continue;
                }
	    }
	    cv::Mat src = cv::imread(sample.getColorImagePath());
            int imgWidth = src.size().width;
            int imgHeight = src.size().height;

	    double xMin = it->region.x / imgWidth;
            double yMin = it->region.y / imgHeight;
            double xMax = (it->region.x + it->region.width) / imgWidth;
            double yMax = (it->region.y + it->region.height) / imgHeight;
            double confidence_score = it->confidence_score;
	    out << sampleId << "," << "xclick" << "," << classId << "," << confidence_score << "," << xMin << "," << xMax << "," << yMin << "," << yMax << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << std::endl;
	}

    }
    if (!out.good()) throw std::runtime_error ("Can't write to the file!");
}
