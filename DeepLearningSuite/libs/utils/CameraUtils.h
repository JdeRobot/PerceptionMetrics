//
// Created by frivas on 4/04/17.
//

#ifndef JDEROBOT_CAMERAUTILS_H
#define JDEROBOT_CAMERAUTILS_H


#include <opencv2/core/core.hpp>
#include <jderobot/image.h>
#include <camera.h>

class CameraUtils {
public:
    static cv::Mat getImageFromCameraProxy(jderobot::ImageDataPtr dataPtr);
    static std::string negotiateDefaultFormat(jderobot::CameraPrx prx,const std::string& definedFormat );
    static bool compressImage(const cv::Mat& image, unsigned char** compressed_data,unsigned long& compress_len);
    static jderobot::ImageDataPtr convert(const cv::Mat& image);
};


#endif //JDEROBOT_CAMERAUTILS_H
