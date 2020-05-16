//
// Created by frivas on 17/11/16.
//

#include "DepthForegroundSegmentator.h"



DepthForegroundSegmentator::DepthForegroundSegmentator(bool filterActive):filterActive(filterActive) {

#if  CV_MAJOR_VERSION == 3 || CV_MAJOR_VERSION == 4
    std::cerr << "OpenCV 3 is not working with doubles foreground segmentation" << std::endl;
    throw "Opencv v3 is not supported";
#else
#endif

    if (this->filterActive){
        this->filter= boost::shared_ptr<jderobot::DepthFilter>(new jderobot::DepthFilter());
    }

    defaultLearningRate=0.0001;
    minBlobArea=400;
}


std::vector<std::vector<cv::Point>> DepthForegroundSegmentator::process(const cv::Mat &image) {
    cv::Mat localImage = image.clone();

    cv::imshow("localImage", localImage);
    if (this->filterActive){
        cv::Mat temp;
        this->filter->update(localImage,temp);
        temp.copyTo(localImage);
    }

    std::vector<cv::Mat> layers;
    cv::split(localImage, layers);

    cv::Mat distance(localImage.rows, localImage.cols, CV_32FC1,cv::Scalar(0,0,0)); // muestreada
    cv::Mat realDistance(localImage.rows, localImage.cols, CV_32FC1); //distancia real

    //discretizamos la imagen de profundidad

    int val=0;
    for (int x=0; x< layers[1].cols ; x++){
        for (int y=0; y<layers[1].rows; y++){
            float d=((int)layers[1].at<unsigned char>(y,x)<<8)|(int)layers[2].at<unsigned char>(y,x);
            distance.at<float>(y,x) = float(floor((pow(d,1./4.)/10)*1600));
            realDistance.at<float>(y,x) = d;
            val++;
        }
    }
    cv::Mat normDistanceFloat;
    cv::normalize(distance,normDistanceFloat,0, 255, cv::NORM_MINMAX, CV_32F);
    cv::Mat normDistance;
    normDistanceFloat.convertTo(normDistance,CV_8UC1);


    cv::Mat normRealDistanceFloat;
    cv::normalize(realDistance,normRealDistanceFloat,0, 255, cv::NORM_MINMAX, CV_32F);
    cv::Mat normRealDistance;
    normRealDistanceFloat.convertTo(normRealDistance,CV_8UC1);

    cv::imshow("distance", normDistance);
    cv::imshow("realDistance", normRealDistance);



    if (!this->bg){
#if  CV_MAJOR_VERSION == 3 || CV_MAJOR_VERSION == 4
        this->bg= cv::createBackgroundSubtractorMOG2();
//        this->bg= cv::createBackgroundSubtractorKNN(500,20000);
        bg->apply(distance,fore,defaultLearningRate);
#else
        this->bg=new cv::BackgroundSubtractorMOG2();
        bg->operator()(distance,fore,defaultLearningRate);
#endif

    }
#if  CV_MAJOR_VERSION == 3 || CV_MAJOR_VERSION == 4
    bg->apply(distance,fore,defaultLearningRate);
#else
    bg->operator()(distance,fore,defaultLearningRate);
#endif


    cv::imshow("fore",fore);
    cv::Mat back(240,320,CV_8UC3,cv::Scalar(0,0,0));
    cv::erode(fore,fore,cv::Mat());
    cv::dilate(fore,fore,cv::Mat());
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(fore,contours, hierarchy,cv::RETR_TREE,cv::CHAIN_APPROX_SIMPLE);




    cv::Mat dst1=cv::Mat(distance.size(), CV_8UC1,cv::Scalar(0));

    std::vector<std::vector<cv::Point>> goodContours;
    int cCounter1=0;
    int idx = 0;
    if (contours.size() != 0){
        for( ; idx >= 0; idx = hierarchy[idx][0] )
        {
            //area minima!!!
            double area0 = contourArea(contours[idx]);
            //UMBRAL DE AREA
            if (area0< this->minBlobArea)
                continue;
            std::vector<cv::Point> a = contours[idx];
            goodContours.push_back(a);
            /*cv::Scalar color( 255);
            cv::drawContours( dst1, contours, idx, color, CV_FILLED, 8, hierarchy );
            std::cout << "something detected" << std::endl;*/
        }
    }

    return  goodContours;

}



cv::Mat DepthForegroundSegmentator::process2(const cv::Mat &image) {
   /* cv::Mat localImage = image.clone();

    std::cout << "size: " << localImage.size() << std::endl;
    cv::imshow("localImage", localImage);
    if (this->filterActive){
        cv::Mat temp;
        this->filter->update(localImage,temp);
        temp.copyTo(localImage);
    }

    std::vector<cv::Mat> layers;
    cv::split(localImage, layers);

    cv::Mat distance(localImage.rows, localImage.cols, CV_32FC1,cv::Scalar(0,0,0)); // muestreada
    cv::Mat realDistance(localImage.rows, localImage.cols, CV_32FC1); //distancia real

    //discretizamos la imagen de profundidad

    std::cout << "size: " << layers[1].cols << ", " << layers[1].rows << std::endl;
    int val=0;
    for (int x=0; x< layers[1].cols ; x++){
        for (int y=0; y<layers[1].rows; y++){
            float d=((int)layers[1].at<unsigned char>(y,x)<<8)|(int)layers[2].at<unsigned char>(y,x);
            distance.at<float>(y,x) = float(floor((pow(d,1./4.)/10)*1600));
            realDistance.at<float>(y,x) = d;
            val++;
        }
    }
    cv::Mat normDistanceFloat;
    cv::normalize(distance,normDistanceFloat,0, 255, cv::NORM_MINMAX, CV_32F);
    cv::Mat normDistance;
    normDistanceFloat.convertTo(normDistance,CV_8UC1);


    cv::Mat normRealDistanceFloat;
    cv::normalize(realDistance,normRealDistanceFloat,0, 255, cv::NORM_MINMAX, CV_32F);
    cv::Mat normRealDistance;
    normRealDistanceFloat.convertTo(normRealDistance,CV_8UC1);

    cv::imshow("distance", normDistance);
    cv::imshow("realDistance", normRealDistance);



    if (!this->bg){
        this->bg= cv::createBackgroundSubtractorMOG2();
        //this->bg= cv::createBackgroundSubtractorKNN(500,20000);
        bg->apply(distance,fore,defaultLearningRate);
    }
    bg->apply(normRealDistance,fore,defaultLearningRate);

    std::cout << "size: " << fore.size() << std::endl;


    cv::imshow("fore",fore);
    cv::waitKey(1);

    cv::Mat back(240,320,CV_8UC3,cv::Scalar(0,0,0));

    cv::erode(fore,fore,cv::Mat());
    cv::dilate(fore,fore,cv::Mat());
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(fore,contours, hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE);




    cv::Mat dst1=cv::Mat(distance.size(), CV_8UC1,cv::Scalar(0));
    std::vector<cv::Point2f> mc1( contours.size() );
    int cCounter1=0;
    int idx = 0;
    if (contours.size() != 0){
        for( ; idx >= 0; idx = hierarchy[idx][0] )
        {
            //area minima!!!
            double area0 = contourArea(contours[idx]);
            //UMBRAL DE AREA
            if (area0< this->minBlobArea)
                continue;
            cv::Scalar color( 255);
            cv::drawContours( dst1, contours, idx, color, CV_FILLED, 8, hierarchy );
            std::cout << "something detected" << std::endl;
        }
    }

    return  dst1;*/

}
