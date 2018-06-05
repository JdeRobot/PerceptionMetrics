//
// Created by frivas on 22/01/17.
//

#include "Sample.h"
#include <glog/logging.h>
#include <iomanip>
#include <boost/filesystem/operations.hpp>


Sample::Sample() {
    this->colorImagePath="";
    this->depthImagePath="";
    this->rectRegions=RectRegionsPtr(new RectRegions());
    this->contourRegions=ContourRegionsPtr(new ContourRegions());

}

Sample::Sample(const cv::Mat &colorImage) {
    colorImage.copyTo(this->colorImage);
}

Sample::Sample(const cv::Mat &colorImage, const RectRegionsPtr &rectRegions) {
    this->setColorImage(colorImage);
    this->setRectRegions(rectRegions);
}

Sample::Sample(const cv::Mat &colorImage, const ContourRegionsPtr &contourRegions) {
    this->setColorImage(colorImage);
    this->setContourRegions(contourRegions);
}

Sample::Sample(const cv::Mat &colorImage, const RectRegionsPtr &rectRegions, const ContourRegionsPtr &contourRegions) {
    this->setColorImage(colorImage);
    this->setRectRegions(rectRegions);
    this->setContourRegions(contourRegions);
}

Sample::Sample(const cv::Mat &colorImage, const cv::Mat &depthImage, const RectRegionsPtr &rectRegions) {
    this->setColorImage(colorImage);
    this->setDepthImage(depthImage);
    this->setRectRegions(rectRegions);
}

Sample::Sample(const cv::Mat &colorImage, const cv::Mat &depthImage, const ContourRegionsPtr &contourRegions) {
    this->setColorImage(colorImage);
    this->setDepthImage(depthImage);
    this->setContourRegions(contourRegions);
}

Sample::Sample(const cv::Mat &colorImage, const cv::Mat &depthImage, const RectRegionsPtr &rectRegions,
               const ContourRegionsPtr &contourRegions) {
    this->setColorImage(colorImage);
    this->setDepthImage(depthImage);
    this->setRectRegions(rectRegions);
    this->setContourRegions(contourRegions);

}

void Sample::setColorImage(const cv::Mat &image) {
    image.copyTo(this->colorImage);
}

void Sample::setDepthImage(const cv::Mat &image) {
    image.copyTo(this->depthImage);
}

void Sample::setRectRegions(const RectRegionsPtr &regions) {
    this->rectRegions=regions;
}

void Sample::setContourRegions(const ContourRegionsPtr &regions) {
    this->contourRegions=regions;
}

RectRegionsPtr Sample::getRectRegions() const{
    return this->rectRegions;
}

ContourRegionsPtr Sample::getContourRegions() {
    return this->contourRegions;
}

std::string Sample::getColorImagePath() const{
    if (this->colorImagePath.empty())
        throw std::invalid_argument("Color Image Path not set in this Sample");

    return this->colorImagePath;
}

std::string Sample::getDepthImagePath() const{
    if (this->depthImagePath.empty())
        throw std::invalid_argument("Depth Image Path not set in this Sample");

    return this->depthImagePath;
}

cv::Mat Sample::getColorImage() const{
    if (this->colorImage.empty()) {
        cv::Mat image = cv::imread(this->colorImagePath);
        return image;
    }
    else
        return this->colorImage.clone();
}

cv::Mat Sample::getDepthImage() const{
    if (this->depthImage.empty()) {
        cv::Mat image = cv::imread(this->depthImagePath);
        return image;
    }
    else
        return this->depthImage.clone();
}

Sample::Sample(const std::string &path, const std::string &id,bool loadDepth) {
    this->colorImagePath=path + "/"  + id + ".png";

    if (boost::filesystem::exists(boost::filesystem::path(path + "/" + id + ".json")))
        this->rectRegions=RectRegionsPtr(new RectRegions(path + "/" + id + ".json"));
    else{
        LOG(ERROR) << "Error " + id + " sample has not associated detection";
    }

    if (boost::filesystem::exists(boost::filesystem::path(path + "/" + id + "-region.json")))
        this->contourRegions=ContourRegionsPtr(new ContourRegions(path + "/" + id + "-region.json"));

    if (loadDepth) {
        this->depthImagePath=path + "/" + id + "-depth.png";
    }
}

cv::Mat Sample::getSampledColorImage() const{
    cv::Mat image = this->getColorImage();
    if (this->rectRegions)
        this->rectRegions->drawRegions(image);
    if (this->contourRegions)
        this->contourRegions->drawRegions(image);
    return image;
}

cv::Mat Sample::getSampledDepthImage() const{
    cv::Mat image =this->getDepthImage();
    if (this->rectRegions)
        this->rectRegions->drawRegions(image);
    if (this->contourRegions)
        this->contourRegions->drawRegions(image);
    return image;
}

void Sample::save(const std::string &outPath, int id) {
    std::stringstream ss ;
    ss << std::setfill('0') << std::setw(5) << id;
    this->save(outPath,ss.str());

}

void Sample::save(const std::string &outPath, const std::string &filename) {


    if (this->colorImage.empty()){
        if (!this->colorImagePath.empty())
            if (boost::filesystem::exists(boost::filesystem::path(this->colorImagePath))) {
                cv::Mat image = cv::imread(this->colorImagePath);
                cv::imwrite(outPath + "/" + filename + ".png", image);
            }
    }
    else
        cv::imwrite(outPath + "/" + filename + ".png",this->colorImage);

    if (this->depthImage.empty()){
        if (boost::filesystem::exists(boost::filesystem::path(this->depthImagePath))) {
            cv::Mat image = cv::imread(this->depthImagePath);
            cv::imwrite(outPath + "/" + filename + "-depth.png", image);
        }
    }
    else
        cv::imwrite(outPath + "/" + filename + "-depth.png", depthImage);

    if(rectRegions)
        rectRegions->saveJson(outPath + "/" + filename + ".json");
    if (contourRegions)
        contourRegions->saveJson(outPath + "/" + filename + "-region.json");
}

void Sample::save(const std::string &outPath) {
    if (this->sampleID.size() != 0 ){
        this->save(outPath,this->sampleID);
    }
    else{
        LOG(ERROR) << "No sample id is defined, this sample will not be saved";
    }

}


bool Sample::isDepthImageValid() {
  return !this->depthImage.empty();
}

bool Sample::isValid() {
    return !this->colorImage.empty();
}

void Sample::filterSamplesByID(std::vector<std::string> filteredIDS) {
    if (this->rectRegions)
        this->rectRegions->filterSamplesByID(filteredIDS);
    if (contourRegions)
        this->contourRegions->filterSamplesByID(filteredIDS);
}

void Sample::setColorImage(const std::string &imagePath) {
    this->colorImagePath=imagePath;
}

void Sample::setDepthImage(const std::string &imagePath) {
    this->depthImagePath=imagePath;
}

void Sample::setSampleID(const std::string &sampleID) {
    this->sampleID=sampleID;
}

std::string Sample::getSampleID() {
    return this->sampleID;
}

Sample::~Sample() {
    if (!this->colorImage.empty()){
        this->colorImage.release();
    }
    if (this->depthImage.empty()){
        this->depthImage.release();
    }

}

cv::Mat Sample::getDeptImageGrayRGB() const {
    cv::Mat image = this->getDepthImage();
    std::vector<cv::Mat> imageVector;
    cv::split(image,imageVector);

    std::vector<cv::Mat> grayRGB_vector;
    grayRGB_vector.push_back(imageVector[0]);
    grayRGB_vector.push_back(imageVector[0]);
    grayRGB_vector.push_back(imageVector[0]);

    cv::Mat grayRGB;
    cv::merge(grayRGB_vector,grayRGB);
    return grayRGB;

}

cv::Mat Sample::getDepthColorMapImage(double alpha, double beta) const {
    cv::Mat image = getDepthImage();
    double minVal, maxVal;

    minMaxLoc( image, &minVal, &maxVal );

    cv::Mat mask;
    cv::threshold(image, mask, maxVal - 1, 255, cv::THRESH_BINARY_INV);
    mask.convertTo(mask, CV_8UC1);

    image.convertTo(image, CV_8UC1, alpha, beta);

    cv::Mat colorMappedDepth;
    cv::applyColorMap(image, image, cv::COLORMAP_RAINBOW);
    image.copyTo(colorMappedDepth, mask);

    return colorMappedDepth;;
}

cv::Mat Sample::getSampledDepthColorMapImage(double alpha, double beta) const {
    cv::Mat image = getDepthColorMapImage(alpha, beta);
    if (this->rectRegions)
        this->rectRegions->drawRegions(image);
    if (this->contourRegions)
        this->contourRegions->drawRegions(image);
    return image;
}
