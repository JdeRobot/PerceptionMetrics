//
// Created by frivas on 22/11/16.
//

#include "BoundingValidator.h"

BoundingValidator::BoundingValidator(const cv::Mat &image_in,double scale) {
    this->image=image_in.clone();
    this->scale=scale;
//    cv::cvtColor(this->image, this->image, CV_RGB2BGR);
    cv::resize(this->image,this->image,cv::Size(), scale,scale);
    clicked=false;
    movementType = BoundingRectGuiMover::NONE;

}

bool BoundingValidator::validate(std::vector<cv::Point> &bounding, cv::Rect_<double>& validatedBound, int& key) {

    cv::Rect_<double> boundingRectangle = cv::boundingRect(bounding);
    validate(boundingRectangle,validatedBound,key);


}





void BoundingValidator::CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    auto rect = (BoundingRectGuiMover*)userdata;
    if  ( event == cv::EVENT_LBUTTONDOWN )
    {
        if (!clicked) {
            movementType = BoundingRectGuiMover::LOCAL_MOVEMENT;
            clicked = true;
            from=cv::Point(x,y);
            tempFrom=cv::Point(x,y);

        }
    }
    else if  ( event == cv::EVENT_LBUTTONUP )
    {
        if (clicked) {
            movementType = BoundingRectGuiMover::NONE;
            clicked = false;
            /*to=cv::Point(x,y);
            std::cout << "moving from: " << from << ", to: " << to << std::endl;
            rect->move(from,to);*/
        }
    }
    else if ( event == cv::EVENT_MOUSEMOVE )
    {
        if (clicked) {
            to=cv::Point(x,y);
//            std::cout << "moving from: " << from << ", to: " << to << std::endl;
            rect->move(tempFrom,to,movementType);
            tempFrom=cv::Point(x,y);


        }
    }
    else if (event == cv::EVENT_MBUTTONDOWN){
        if (!clicked) {
            movementType = BoundingRectGuiMover::GLOBAL_MOVEMENT;
            clicked = true;
            from=cv::Point(x,y);
            tempFrom=cv::Point(x,y);

        }
    }
    else if (event == cv::EVENT_MBUTTONUP){
        movementType = BoundingRectGuiMover::NONE;
        if (clicked) {
            clicked = false;
        }
    }
}

bool BoundingValidator::validate(const cv::Rect_<double> &bounding, cv::Rect_<double> &validatedBound, int &key) {


    cv::Rect_<double> scaledBounding((int)(bounding.x * this->scale),
                            (int)(bounding.y*this->scale),
                            (int)(bounding.width * this->scale),
                            (int)(bounding.height*this->scale));

    BoundingRectGuiMover rect(scaledBounding);
    std::vector<char> validationKeys;
    validationKeys.push_back('1');
    validationKeys.push_back('2');
    validationKeys.push_back('3');
    char rejectionKey='n';
    key='0';
    while (char(key) != rejectionKey and (std::find(validationKeys.begin(), validationKeys.end(), char(key))== validationKeys.end() )) {
        cv::Mat image2show= this->image.clone();
        cv::rectangle(image2show, rect.getRect(), cv::Scalar(255, 0, 0), 3);
        std::string windowName="Validation window";
        cv::namedWindow(windowName, 1);
        cv::setMouseCallback(windowName, CallBackFunc, &rect);
        cv::imshow(windowName, image2show);
        key=cv::waitKey(1);
        //std::cout << "key: " << char(key) << std::endl;
    }
    validatedBound=rect.getRect(this->scale);


    return std::find(validationKeys.begin(), validationKeys.end(), char(key))!= validationKeys.end();
}

bool BoundingValidator::validateNDetections(std::vector<RectRegion> &regions) {
    auto imageInputRects=updateRegionsImage(regions);

    std::string windowName="Validate number of detecions";

    char key='p';
    cv::Rect_<double> rect;
    char rejectionKey='q';
    while ((key != ' ') and (key != rejectionKey)){
        cv::Mat image2show= imageInputRects.clone();

        cv::setMouseCallback(windowName, CallBackFuncNumberDetections, &rect);
        if (rect != cv::Rect_<double>()){
            cv::rectangle(image2show,rect,cv::Scalar(0,255,0),int(this->scale));
            if (!clicked){
                cv::Rect_<double> newScaledRect=cv::Rect_<double>(int(rect.x/scale),
                                                int(rect.y/scale),
                                                int(rect.width/scale),
                                                int(rect.height/scale));
                RectRegion newRegion(newScaledRect,"person");
                regions.push_back(newRegion);
                imageInputRects=updateRegionsImage(regions);
                rect=cv::Rect_<double>();
            }
        }
        cv::imshow(windowName,image2show);
        key=(char)cv::waitKey(100);
    }
    if (key == 'q'){
        regions.clear();
    }
    cv::destroyWindow(windowName);
    return false;
}



void BoundingValidator::CallBackFuncNumberDetections(int event, int x, int y, int flags, void* userdata)
{
    auto rect = (cv::Rect_<double>*)userdata;
    if  ( event == cv::EVENT_LBUTTONDOWN )
    {
        if (!clicked) {
            movementType = BoundingRectGuiMover::LOCAL_MOVEMENT;
            clicked = true;
            from=cv::Point(x,y);
            tempFrom=cv::Point(x,y);
            rect->x=x;
            rect->y=y;
            rect->width=1;
            rect->height=1;

        }
    }
    else if  ( event == cv::EVENT_LBUTTONUP )
    {
        if (clicked) {
            movementType = BoundingRectGuiMover::NONE;
            clicked = false;
            rect->width=to.x - from.x;
            rect->height=to.y - from.y;
        }
    }
    else if ( event == cv::EVENT_MOUSEMOVE )
    {
        if (clicked) {
            to=cv::Point(x,y);
            rect->width=to.x - from.x;
            rect->height=to.y - from.y;
        }
    }

}

cv::Mat BoundingValidator::updateRegionsImage(const std::vector<RectRegion> &regions) {
    cv::Mat imageInputRects= this->image.clone();
    for (auto it:regions) {
        auto bounding=it.region;
        cv::Rect_<double> scaledBounding(int(bounding.x * this->scale),
                                int(bounding.y*this->scale),
                                int(bounding.width * this->scale),
                                int(bounding.height*this->scale));
        cv::rectangle(imageInputRects, scaledBounding, cv::Scalar(0, 255, 255), 3);
    }
    return imageInputRects;
}
