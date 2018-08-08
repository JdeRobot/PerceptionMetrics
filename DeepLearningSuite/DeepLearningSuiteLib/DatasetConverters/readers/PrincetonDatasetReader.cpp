//
// Created by frivas on 29/07/17.
//

#include <bitset>
#include "PrincetonDatasetReader.h"
#include <Utils/StringHandler.h>
#include <boost/filesystem/operations.hpp>
#include <fstream>
#include <Utils/PathHelper.h>
#include <glog/logging.h>
#include <sstream>
#include <map>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <Utils/JsonHelper.h>
#include <Utils/DepthUtils.h>


PrincetonDatasetReader::PrincetonDatasetReader(const std::string &path, const std::string &classNamesFile,const bool imagesRequired):DatasetReader(imagesRequired) {
    this->classNamesFile=classNamesFile;
    appendDataset(path);
}

bool PrincetonDatasetReader::appendDataset(const std::string &datasetPath, const std::string &datasetPrefix) {
    std::string framesData=PathHelper::concatPaths(datasetPath,"frames.json");
    auto boostPath= boost::filesystem::path(framesData);

    if (!boost::filesystem::exists(framesData.c_str())){
        LOG(ERROR) << "Dataset path: " + datasetPath + " does not contain any frames.json file";
    }



    boost::property_tree::ptree pt;
    boost::property_tree::read_json(framesData, pt);
    std::string foo = pt.get<std::string> ("format");

    auto depthTimestamp=JsonHelper::as_vector<int>(pt, "depthTimestamp");
    auto depthFrameID=JsonHelper::as_vector<int>(pt, "depthFrameID");
    auto imageTimestamp=JsonHelper::as_vector<int>(pt, "imageTimestamp");
    auto imageFrameID=JsonHelper::as_vector<int>(pt, "imageFrameID");

    for (size_t i = 0; i < depthTimestamp.size(); i++) {
        LOG(INFO) << "Loading: " << i << " of " << depthTimestamp.size();

        //depth Image
        std::stringstream ssDepth;
        ssDepth << "d-" << depthTimestamp[i] << "-" << depthFrameID[i] << ".png";
        std::string depthImagePath = PathHelper::concatPaths(datasetPath, "depth");
        depthImagePath = PathHelper::concatPaths(depthImagePath, ssDepth.str());
        cv::Mat depthImage = cv::imread(depthImagePath, cv::IMREAD_ANYDEPTH);
        cv::Mat ownDepthImage;
        DepthUtils::mat16_to_ownFormat(depthImage,ownDepthImage);



        //colorImage
        std::stringstream ssColor;
        ssColor << "r-" << imageTimestamp[i] << "-" << imageFrameID[i] << ".png";
        std::string colorImagePath = PathHelper::concatPaths(datasetPath, "rgb");
        colorImagePath = PathHelper::concatPaths(colorImagePath, ssColor.str());
        cv::Mat colorImage = cv::imread(colorImagePath);


        Sample sample;
        sample.setDepthImage(ownDepthImage);
        sample.setColorImage(colorImage);
        samples.push_back(sample);
    }

    printDatasetStats();

    return true;
}
