//
// Created by frivas on 30/07/17.
//

#include <iostream>
#include "DepthUtils.h"
#include <glog/logging.h>
void DepthUtils::mat16_to_ownFormat(const cv::Mat &inputImage, cv::Mat& outImage) {

    double MAX_LENGHT=10000;

    auto imageSize = inputImage.size();
    outImage=cv::Mat(imageSize,CV_8UC3,cv::Scalar(0,0,0));


    cv::Mat evalImage= inputImage.clone();
    evalImage=cv::Scalar(0,0,0);

    for (int y = 0; y < imageSize.height; ++y) {
        for (int x = 0; x < imageSize.width; ++x) {
            uint16_t value = inputImage.at<uint16_t>(y, x);
            value= (value >> 3);
            evalImage.at<uint16_t>(y, x) = value;
            outImage.data[(y*imageSize.width+ x)*3+0] = (float(value)/(float)MAX_LENGHT)*255.;
            outImage.data[(y*imageSize.width+ x)*3+1] = (value)>>8;
            outImage.data[(y*imageSize.width+ x)*3+2] = (value)&0xff;
        }
    }


    double min,max;
    cv::minMaxLoc(inputImage,&min,&max);

    LOG(INFO) << "min: " << min << std::endl;
    LOG(INFO) << "max: " << max << std::endl;

    cv::minMaxLoc(evalImage,&min,&max);

    LOG(INFO) << "min: " << min << std::endl;
    LOG(INFO) << "max: " << max << std::endl;



}

void DepthUtils::spinello_mat16_to_viewable(const cv::Mat &inputImage, cv::Mat& outImage) {

    double min,max;
    cv::minMaxLoc(inputImage,&min,&max);


    if (max > 10000) {              // Swapping bytes

      cv::Mat toswap(inputImage.rows, inputImage.cols, CV_8UC2, inputImage.data);
      cv::Mat merged;

      std::vector<cv::Mat> channels(2);
      cv::split(toswap, channels);
      std::reverse(channels.begin(), channels.end());
      cv::merge(&channels[0], 2, merged);

      merged.addref();
      outImage = cv::Mat(toswap.rows, toswap.cols, CV_16UC1, merged.data);

    } else {
      outImage = inputImage;
    }

}
