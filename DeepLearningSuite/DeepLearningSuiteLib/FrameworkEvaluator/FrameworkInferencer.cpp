//
// Created by frivas on 24/02/17.
//


#include "FrameworkInferencer.h"
#include <glog/logging.h>


Sample FrameworkInferencer::detect(const cv::Mat &image, double confidence_threshold) {
    boost::posix_time::ptime startTime = boost::posix_time::microsec_clock::local_time();
    Sample s = detectImp(image, confidence_threshold);
    boost::posix_time::ptime endTime = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration duration = endTime-startTime;
    long elapsedTime = duration.total_milliseconds();
    durationVector.push_back(elapsedTime);
    return s;
}

int FrameworkInferencer::getMeanDurationTime() {
    int accumulate=0;
    for (auto it = this->durationVector.begin(), end = this->durationVector.end(); it != end; ++it)
        accumulate+=(int)(*it);
    if (this->durationVector.size() ==0)
        return 0;
    else
        return accumulate/(int)this->durationVector.size();
}

FrameworkInferencer::FrameworkInferencer() {

}

FrameworkInferencer::~FrameworkInferencer() {
    LOG(INFO) << "Mean inference time: " << this->getMeanDurationTime() << "(ms)" <<  std::endl;
}
