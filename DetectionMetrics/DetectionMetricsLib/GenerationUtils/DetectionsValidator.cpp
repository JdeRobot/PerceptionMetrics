//
// Created by frivas on 20/01/17.
//

#include <boost/filesystem.hpp>
#include <iomanip>
#include "DetectionsValidator.h"
#include "BoundingValidator.h"
#include <fstream>
#include <boost/lexical_cast.hpp>
#include <glog/logging.h>



DetectionsValidator::DetectionsValidator(const std::string& pathToSave,double scale):validationCounter(0), path(pathToSave),scale(scale) {
    auto boostPath= boost::filesystem::path(this->path);
    if (!boost::filesystem::exists(boostPath)){
        boost::filesystem::create_directories(boostPath);
    }
    else{
        boost::filesystem::directory_iterator end_itr;

        for (boost::filesystem::directory_iterator itr(boostPath); itr!=end_itr; ++itr)
        {
            if ((is_regular_file(itr->status()) && itr->path().extension()==".png") && (itr->path().string().find("-depth") == std::string::npos)) {
                validationCounter++;
            }

        }
        if (this->validationCounter != 0) {
            LOG(WARNING) << "Including samples to an existing dataset, starting with: " +
                            std::to_string(this->validationCounter);
            char confirmation = 'a';
            while (confirmation != 'y' && confirmation != 'n') {
                LOG(INFO) << "Do you want to continue? (y/n) \n";
                std::cin >> confirmation;
            }
            if (confirmation == 'n') {
                LOG(WARNING) << "Exiting";
                exit(1);
            }
        }
    }

}

DetectionsValidator::~DetectionsValidator()=default;



void DetectionsValidator::validate(const cv::Mat& colorImage,const cv::Mat& depthImage, std::vector<std::vector<cv::Point>>& detections){

    cv::Mat mask=cv::Mat(colorImage.size(), CV_8UC1,cv::Scalar(0));

    for (auto it= detections.begin(), end = detections.end(); it != end; ++it){
        int idx= (int)std::distance(detections.begin(),it);
        cv::Scalar color( 150);
        cv::drawContours( mask, detections, idx, color, -1, 8);
    }


    std::vector<cv::Mat> channels;
    cv::split(colorImage,channels);
    cv::Mat colorMask(colorImage.size(),CV_8UC1,cv::Scalar(150));
    colorMask.copyTo(channels[0],mask);
    cv::Mat image2show;
    cv::merge(channels,image2show);


    RectRegionsPtr regions(new RectRegions());
    ContourRegionsPtr cRegions(new ContourRegions());

    BoundingValidator validator(image2show,scale);
    for (auto it : detections){
        cv::Rect_<double> validatedRect;
        int classVal;
        if (validator.validate(it,validatedRect,classVal)){
            std::string validationID;
            if (char(classVal)=='1') {
                validationID = "person";
            }
            else if (char(classVal)=='2')
                validationID="person-falling";
            else if (char(classVal)=='3')
                validationID="person-fall";


            fillRectIntoImageDimensions(validatedRect,colorImage.size());
            LOG(INFO)<<"Validated";
            regions->add(validatedRect,validationID);
            cRegions->add(it,validationID);
        }
        else{
            LOG(INFO)<<"Discarded";
        }
    }

    if (not regions->getRegions().empty()){
        Sample sample(colorImage,depthImage,regions,cRegions);
        sample.save(this->path,this->validationCounter);
        this->validationCounter++;
    }
}

void DetectionsValidator::fillRectIntoImageDimensions(cv::Rect_<double> &rect, const cv::Size size) {
    //check the format x,y -> w.h

    if (rect.width < 0){
        rect.x=rect.x-rect.width;
        rect.width*=-1;
    }
    if (rect.height < 0){
        rect.y=rect.y-rect.height;
        rect.height*=-1;
    }

    if (rect.x + rect.width > size.width){
        rect.width = size.width - rect.x - 1;
    }
    if (rect.y + rect.height > size.height){
        rect.height = size.height - rect.y -1;
    }

    if (rect.x < 0){
        rect.width=rect.width + rect.x;
        rect.x=0;
    }
    if (rect.y <0){
        rect.height = rect.height + rect.y;
        rect.y=0;
    }

}

void DetectionsValidator::validate(const Sample &inputSample) {
    auto rectDetections = inputSample.getRectRegions()->getRegions();
    RectRegionsPtr validatedRegions(new RectRegions());
    cv::Mat initialImage=inputSample.getColorImage().clone();
    cv::imshow("Source Image", initialImage);
    cv::waitKey(100);
    LOG(INFO) << "Number of detections: " << rectDetections.size() << std::endl;

    BoundingValidator validatorNumber(initialImage, this->scale);


    validatorNumber.validateNDetections(rectDetections);


    for (auto it= rectDetections.begin(), end=rectDetections.end(); it != end; ++it){
        //draw all detections
        cv::Mat currentTestImage=initialImage.clone();

        for (auto it2= rectDetections.begin(), end2=rectDetections.end(); it2 != end2; ++it2) {
            if (it2== it)
                continue;
            cv::rectangle(currentTestImage,it2->region,cv::Scalar(255,255,0));
            cv::imshow("Source Image", currentTestImage);
            cv::waitKey(100);

        }
        BoundingValidator validator(currentTestImage, this->scale);



        cv::Rect_<double> validatedRect;
        int classVal;
        if (validator.validate(it->region,validatedRect,classVal)){
            std::string validationID;
            if (char(classVal)=='1') {
                validationID = "person";
            }
            else if (char(classVal)=='2')
                validationID="person-falling";
            else if (char(classVal)=='3')
                validationID="person-fall";


            fillRectIntoImageDimensions(validatedRect,inputSample.getColorImage().size());
            LOG(INFO)<<"Validated";
            validatedRegions->add(validatedRect,validationID);
        }
        else{
            LOG(INFO)<<"Discarded";
        }
    }

    if (not validatedRegions->getRegions().empty()){
        Sample sample(inputSample.getColorImage(),inputSample.getDepthImage(),validatedRegions);
        sample.save(this->path,this->validationCounter);
        this->validationCounter++;
    }
}
