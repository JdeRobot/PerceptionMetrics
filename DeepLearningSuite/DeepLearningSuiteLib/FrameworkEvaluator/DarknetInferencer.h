//
// Created by frivas on 31/01/17.
//

#ifndef SAMPLERGENERATOR_DARKNETEVALUATOR_H
#define SAMPLERGENERATOR_DARKNETEVALUATOR_H

#include "Wrappers/DarknetAPI.h"
#include <boost/shared_ptr.hpp>
#include "FrameworkInferencer.h"

class DarknetInferencer: public FrameworkInferencer {
public:
    DarknetInferencer(const std::string& netConfig, const std::string& netWeights, const std::string& classNamesFile);
    Sample detectImp(const cv::Mat& image, double confidence_threshold);

private:
    std::string netConfig;
    std::string netWeights;
    boost::shared_ptr<DarknetAPI> cnn;
};


typedef boost::shared_ptr<DarknetInferencer> DarknetInferencerPtr;



#endif //SAMPLERGENERATOR_DARKNETEVALUATOR_H
