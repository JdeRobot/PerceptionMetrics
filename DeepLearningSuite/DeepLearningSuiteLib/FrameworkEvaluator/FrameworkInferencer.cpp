//
// Created by frivas on 24/02/17.
//


#include "FrameworkInferencer.h"
#include <glog/logging.h>


Sample FrameworkInferencer::detect(const cv::Mat &image, double confidence_threshold) {
    // Timestamp just before we start our detection
    boost::posix_time::ptime startTime = boost::posix_time::microsec_clock::local_time();
    // Start detection
    Sample s = detectImp(image, confidence_threshold);
    // Timestamp after we finish our detection.
    boost::posix_time::ptime endTime = boost::posix_time::microsec_clock::local_time();
    // total duration(for one detection): Time after the process is completed - Time before the process is completed.
    boost::posix_time::time_duration duration = endTime-startTime;
    // convert the above duration into total milliseconds taken.
    long elapsedTime = duration.total_milliseconds();
    // Store the elapsedTime in a vector which will later be used to calculate mean time.
    durationVector.push_back(elapsedTime);
    // return the Sample.
    return s;
}

int FrameworkInferencer::getMeanDurationTime() {
    // stores the total time taken
    int accumulate=0;
    // iterate over the entire duration vector.
    for (auto it = this->durationVector.begin(), end = this->durationVector.end(); it != end; ++it)
        accumulate+=(int)(*it);
    // If the duration vector is empty return 0.
    if (this->durationVector.size() ==0)
        return 0;
    // Else return the average time taken.
    else
        return accumulate/(int)this->durationVector.size();
}

FrameworkInferencer::FrameworkInferencer() {

}

/*
    After inferencing log the information regarding the mean time taken and
    the inferencer.
*/
FrameworkInferencer::~FrameworkInferencer() {
    LOG(INFO) << "Mean inference time: " << this->getMeanDurationTime() << "(ms)" <<  std::endl;
}
