//
// Created by frivas on 4/02/17.
//

// A generic framework which encapsulates all other frameworks in this.
#ifndef SAMPLERGENERATOR_GENERICINFERENCER_H
#define SAMPLERGENERATOR_GENERICINFERENCER_H

#include "FrameworkInferencer.h"

// If Darknet is present include the header files of Darknet inferencer.
#ifdef DARKNET_ACTIVE
#include <FrameworkEvaluator/DarknetInferencer.h>
#endif

// Include the header files of tensorflow and keras inferencers.
#include <FrameworkEvaluator/TensorFlowInferencer.h>
#include <FrameworkEvaluator/KerasInferencer.h>

// If Caffe is present include it's header files as well.
#ifdef ENABLE_DNN_CAFFE
#include <FrameworkEvaluator/CaffeInferencer.h>
#endif

// Inferencer can be implemented using any one of the following frameworks.
enum INFERENCER_IMPLEMENTATIONS{INF_YOLO, INF_TENSORFLOW, INF_KERAS, INF_CAFFE};


class GenericInferencer {
public:
    // Constructor function.
    GenericInferencer(const std::string& netConfig, const std::string& netWeights, const std::string& classNames, const std::string& implementation, std::map<std::string, std::string>* inferencerParamsMap = NULL);
    // Get the inferencer which we have selected to use.
    FrameworkInferencerPtr getInferencer();
    // Get all the availableImplementations.
    static std::vector<std::string> getAvailableImplementations();

private:
    INFERENCER_IMPLEMENTATIONS imp;
#ifdef DARKNET_ACTIVE
    DarknetInferencerPtr darknetInferencerPtr;
#endif

    TensorFlowInferencerPtr tensorFlowInferencerPtr;

    KerasInferencerPtr kerasInferencerPtr;

#ifdef ENABLE_DNN_CAFFE
    CaffeInferencerPtr caffeInferencerPtr;
#endif

    std::vector<std::string> availableImplementations;

    static void configureAvailablesImplementations(std::vector<std::string>& data);
    INFERENCER_IMPLEMENTATIONS getImplementation(const std::string& inferencerImplementation);
};

typedef  boost::shared_ptr<GenericInferencer> GenericInferencerPtr;


#endif //SAMPLERGENERATOR_GENERICINFERENCER_H
