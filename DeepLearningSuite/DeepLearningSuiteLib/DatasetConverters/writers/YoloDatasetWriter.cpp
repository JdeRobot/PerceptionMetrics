//
// Created by frivas on 22/01/17.
//

#include "YoloDatasetWriter.h"
#include "DatasetConverters/ClassTypeOwn.h"
#include <iomanip>
#include <fstream>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <glog/logging.h>


YoloDatasetWriter::YoloDatasetWriter(const std::string &outPath, DatasetReaderPtr &reader, bool overWriteclassWithZero):DatasetWriter(outPath,reader),overWriteclassWithZero(overWriteclassWithZero){

    this->fullImagesPath=boost::filesystem::absolute(boost::filesystem::path(outPath + "/JPEGImages")).string();
    this->fullLabelsPath=boost::filesystem::absolute(boost::filesystem::path(outPath + "/labels")).string();


    auto boostImages= boost::filesystem::path(fullImagesPath);
    if (!boost::filesystem::exists(boostImages)){
        boost::filesystem::create_directories(boostImages);
    }
    auto boostLabels= boost::filesystem::path(fullLabelsPath);
    if (!boost::filesystem::exists(boostLabels)){
        boost::filesystem::create_directories(boostLabels);
    }


    LOG(INFO) << "FullImagesPath: " << this->fullImagesPath << std::endl;
    LOG(INFO) << "FullLabelsPath: " << this->fullImagesPath << std::endl;

}

void YoloDatasetWriter::process(bool writeImages, bool useDepth) {
    Sample sample;
    int id=0;
    unsigned int skip_count = 0;


    std::ofstream sampleFile(this->outPath + "/sample.txt");

    while (reader->getNextSample(sample)){
        auto boundingBoxes = sample.getRectRegions()->getRegions();
        std::stringstream ssID ;
        ssID << std::setfill('0') << std::setw(5) << id;
        std::string imageFilePath= this->fullImagesPath + "/" + ssID.str() + ".jpg";
        sampleFile << imageFilePath << std::endl;

        std::string labelFilePath= this->fullLabelsPath + "/" + ssID.str() + ".txt";
        std::ofstream out(labelFilePath);

        cv::Mat image;
        if (writeImages) {
            if (useDepth) {
                image= sample.getDepthImage();
            } else {
                image= sample.getColorImage();
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
            double x = it->region.x;
            double y = it->region.y;
            double w = it->region.width;
            double h = it->region.height;

            double confidence_score = it->confidence_score;

            if ((w + x) > image.size().width){
                w =  image.size().width - 1 - x;
            }
            if ((h + y) > image.size().height){
                h = image.size().height - 1 - y;
            }

            int classId;
            if (overWriteclassWithZero)
                classId=0;
            else {
                ClassTypeOwn typeConverter(it->classID);
                classId = typeConverter.getClassID();
            }
            std::stringstream boundSrt;
            boundSrt << classId <<" " <<  (it->region.x + w/2.0) / (double)image.size().width << " " << (it->region.y + h/2.0) / (double)image.size().height << " " << w / image.size().width << " " << h / image.size().height;
//            std::cout << boundSrt.str() << std::endl;
            out << boundSrt.str() << std::endl;
        }
        out.close();
        id++;

    }
    sampleFile.close();

}
