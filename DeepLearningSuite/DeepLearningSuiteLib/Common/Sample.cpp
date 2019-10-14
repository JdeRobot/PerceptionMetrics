//
// Created by frivas on 22/01/17.
//

#include "Sample.h"
#include <glog/logging.h>
#include <iomanip>
#include <boost/filesystem/operations.hpp>


// Constructor which creates many variables
Sample::Sample() {
    this->colorImagePath="";
    this->depthImagePath="";
    this->rectRegions=RectRegionsPtr(new RectRegions());
    this->contourRegions=ContourRegionsPtr(new ContourRegions());
    this->rleRegions=RleRegionsPtr(new RleRegions());
}

//Constructor
Sample::Sample(const cv::Mat &colorImage) {
    colorImage.copyTo(this->colorImage);
}

//Constructor
Sample::Sample(const cv::Mat &colorImage, const RectRegionsPtr &rectRegions) {
    this->setColorImage(colorImage);
    this->setRectRegions(rectRegions);
}

//Constructor
Sample::Sample(const cv::Mat &colorImage, const ContourRegionsPtr &contourRegions) {
    this->setColorImage(colorImage);
    this->setContourRegions(contourRegions);
}

//Constructor
Sample::Sample(const cv::Mat &colorImage, const RectRegionsPtr &rectRegions, const ContourRegionsPtr &contourRegions) {
    this->setColorImage(colorImage);
    this->setRectRegions(rectRegions);
    this->setContourRegions(contourRegions);
}

//Constructor
Sample::Sample(const cv::Mat &colorImage, const cv::Mat &depthImage, const RectRegionsPtr &rectRegions) {
    this->setColorImage(colorImage);
    this->setDepthImage(depthImage);
    this->setRectRegions(rectRegions);
}

//Constructor
Sample::Sample(const cv::Mat &colorImage, const cv::Mat &depthImage, const ContourRegionsPtr &contourRegions) {
    this->setColorImage(colorImage);
    this->setDepthImage(depthImage);
    this->setContourRegions(contourRegions);
}

//Constructor
Sample::Sample(const cv::Mat &colorImage, const cv::Mat &depthImage, const RectRegionsPtr &rectRegions,
               const ContourRegionsPtr &contourRegions) {
    this->setColorImage(colorImage);
    this->setDepthImage(depthImage);
    this->setRectRegions(rectRegions);
    this->setContourRegions(contourRegions);

}

// Set the dimensions of the sample
void Sample::setSampleDims(const int width, const int height) {
    this->width = width;
    this->height = height;
}

// Set the image colorImage member to the passed image
void Sample::setColorImage(const cv::Mat &image) {
    image.copyTo(this->colorImage);
}

void Sample::clearColorImage() {             // For better memory management
    if (!this->colorImage.empty())
        this->colorImage.release();
}

void Sample::clearDepthImage() {            // For better memory management
    if (!this->depthImage.empty())
        this->depthImage.release();
}

// Set the depthImage
void Sample::setDepthImage(const cv::Mat &image) {
    image.copyTo(this->depthImage);
}

// Set the RectRegions member to the new regions
void Sample::setRectRegions(const RectRegionsPtr &regions) {
    this->rectRegions=regions;
}

void Sample::setContourRegions(const ContourRegionsPtr &regions) {
    this->contourRegions=regions;
}

void Sample::setRleRegions(const RleRegionsPtr& regions) {
    this->rleRegions=regions;
}


int Sample::getSampleWidth() const {
    if (this->width != -1)
        return this->width;

    if (!this->getColorImage().empty())
        return this->getColorImage().cols;

    if (!this->getDepthImage().empty())
        return this->getDepthImage().cols;

    return -1;
}

int Sample::getSampleHeight() const {
    if (this->height != -1)
        return this->height;

    if (!this->getColorImage().empty())
        return this->getColorImage().rows;

    if (!this->getDepthImage().empty())
        return this->getDepthImage().rows;

    return -1;
}

RectRegionsPtr Sample::getRectRegions() const{
    return this->rectRegions;
}

ContourRegionsPtr Sample::getContourRegions() const{
    return this->contourRegions;
}

RleRegionsPtr Sample::getRleRegions() const{
    return this->rleRegions;
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

// Constructor
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
    if (this->rleRegions)
        this->rleRegions->drawRegions(image);
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

    bool ifRegions = false;
    if(!rectRegions->getRegions().empty()) {
        rectRegions->saveJson(outPath + "/" + filename + ".json");
        ifRegions = true;
    }
    if (!contourRegions->getRegions().empty()) {
        contourRegions->saveJson(outPath + "/" + filename + "-region.json");
        ifRegions = true;
    }

    if (!ifRegions)
        LOG(WARNING) << "Both ContourRegions and Rect Regions are not present, hence not saving any regions for Sample: " << this->sampleID;
}

void Sample::save(const std::string &outPath) {
    if (this->sampleID.size() != 0 ){
        this->save(outPath,this->sampleID);
    }
    else{
        LOG(ERROR) << "No sample id is defined, this sample will not be saved";
    }

}

// Print the detections
void Sample::print() {
    LOG(INFO) << "Printing Regions with Classes" << '\n';
    std::vector<RectRegion> regionsToPrint = this->rectRegions->getRegions();
    for (auto it = regionsToPrint.begin(); it != regionsToPrint.end(); it++) {
        LOG(INFO) << "Class: " << it->classID << '\n';
        LOG(INFO) << "Confidence: " << it->confidence_score << '\n';
        LOG(INFO) << "uniqObjectID" << it->uniqObjectID <<'\n';
        LOG(INFO) << "BBOX" << it->region.x << it->region.y << it->region.width << it->region.height << '\n';
    }
}


// To get a positive number 
int mod(int test){
    if(test<0)
      return -test;
    return test;
}

// Adds detections to the frame
void Sample::AddDetection(cv::Rect &detection,std::vector<std::string> *classNames){
  RectRegion temp;
  temp.region.x = detection.x;
  temp.region.y = detection.y;
  temp.region.width = detection.width;
  temp.region.height = detection.height;
  // To get the class names and probability from the user.
  AddClass *a = new AddClass();
  a->SetInit(classNames,&temp.classID,&temp.confidence_score);
  a->show();
  a->wait();
  if(temp.classID.length())
    this->rectRegions->regions.push_back(temp);
}

// Adjust the bounding boxes , and if successfully changed any boundary return true
// else false.
bool Sample::AdjustBox(int x, int y){
  // x and y are current mouse pointer positions
  // Find the corner which is nearer to the mouse pointer
      for (auto it = this->rectRegions->regions.begin(); it != this->rectRegions->regions.end(); it++) {
          if(mod(it->region.x-x)<20 && mod(it->region.y-y)<20){
            it->region.width  -= (x-it->region.x);
            it->region.height -= (y-it->region.y);
            it->region.x=x;
            it->region.y=y;
            return true;
          }
          else if(mod(it->region.x+it->region.width-x)<20 && mod(it->region.y-y)<20){
            it->region.width  += (x-(it->region.x+it->region.width));
            it->region.height -= (y-it->region.y);
            it->region.y=y;
            return true;
          }
          else if(mod(it->region.x-x)<20 && mod(it->region.y+it->region.height-y)<20){
            it->region.width  -= (x-it->region.x);
            it->region.height += (y-(it->region.y+it->region.height));
            it->region.x=x;
            return true;
          }
          else if(mod(it->region.x+it->region.width-x)<20 && mod(it->region.y+it->region.height-y)<20){
            it->region.width  += (x-it->region.x-it->region.width);
            it->region.height += (y-it->region.y-it->region.height);
            return true;
          }
      }
      return false;
}

bool Sample::show(const std::string readerImplementation, const std::string windowName, const int waitKey, const bool showDepth) {
    cv::Mat image = this->getSampledColorImage();
    cv::imshow(windowName, image);

    if (showDepth) {

        if (!(this->isDepthImageValid())) {
            LOG(WARNING)<< "Depth Images not available! Please verify your dataset or uncheck 'Show Depth Images'";
            return false;
        }

        cv::Mat depth_color;

        if (readerImplementation == "spinello")
            depth_color = this->getSampledDepthColorMapImage(-0.9345, 1013.17);
        else
            depth_color = this->getSampledDepthColorMapImage();
        cv::imshow("Depth Color Map", depth_color);
    }

    int key = cv::waitKey(waitKey);
    if (char(key) == 'q' || key == 27) {
        cv::destroyWindow(windowName);
        return false;
    }

    return true;

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


// Set if mouse is clicked
void Sample::SetMousy(bool mousy){
  this->mousy = mousy;
}

// Set the current state of mouse
bool Sample::GetMousy(){
  return this->mousy;
}

// This function is used to change the classes of wrongly classified detections
void Sample::SetClassy(int x , int y, std::vector<std::string> *classNames){
  // Check if the user clicked inside certain boundaries
      for (auto it = this->rectRegions->regions.begin(); it != this->rectRegions->regions.end(); it++)
          if(it->nameRect.x<x && it->nameRect.x+it->nameRect.width>x)
            if(it->nameRect.y<y && it->nameRect.y+it->nameRect.height>y){
              LOG(INFO) << "I'm inside rectName" << std::endl;
              LOG(INFO) << "ClassId : " << it->classID <<std::endl;
              // If yes, create a GUI and pop it using which he/she could change it.
              SetClass *w = new SetClass();
              w->SetInit(&it->classID,classNames,&it->classID);
              w->show();
              w->wait();
              break;
            }
}
