//
// Created by frivas on 29/01/17.
//

#ifndef SAMPLERGENERATOR_FRAMEWORKEVALUATOR_H
#define SAMPLERGENERATOR_FRAMEWORKEVALUATOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/shared_ptr.hpp>
#include <Common/Sample.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <algorithm>

class FrameworkInferencer{
public:
    // Constructor function
    FrameworkInferencer();
    // Destructor function
    ~   FrameworkInferencer();
    // Detect objects in a image and return the information stored in a sample.
    Sample detect(const cv::Mat& image, double confidence_threshold);
    // Get the total time taken for inferencing different objects.
    int getMeanDurationTime();
    // Below one will be defined by the child class which inherits this as parent.
    virtual Sample detectImp(const cv::Mat& image, double confidence_threshold) =0;

protected:
    // Path where the class names are stored.
    std::string classNamesFile;

private:
    // This vector stores the time taken to detect an object in an image.
    std::vector<long> durationVector;
};




typedef boost::shared_ptr<FrameworkInferencer> FrameworkInferencerPtr;


#endif //SAMPLERGENERATOR_FRAMEWORKEVALUATOR_H
