//
// Created by frivas on 4/02/17.
//

#include <glog/logging.h>
#include "GenericInferencer.h"

// Process the image using the selected inferener.
GenericInferencer::GenericInferencer(const std::string &netConfig, const std::string &netWeights,const std::string& classNames,
                                     const std::string &implementation, std::map<std::string, std::string>* inferencerParamsMap) {
    // Get all the available inferencers which are present at the user's end.
    configureAvailablesImplementations(this->availableImplementations);

    // Check if the selected inferencer is available or not
    if (std::find(this->availableImplementations.begin(), this->availableImplementations.end(), implementation) != this->availableImplementations.end()){
        // If available , get the inferencer implementation and store it in imp.
        imp = getImplementation(implementation);
        // Inference the image image using the selected inferencer(Currently supports 4 different inferencers).
        switch (imp) {
#ifdef DARKNET_ACTIVE
            case INF_YOLO:
                this->darknetInferencerPtr = DarknetInferencerPtr( new DarknetInferencer(netConfig, netWeights,classNames));
                break;
#endif
            case INF_TENSORFLOW:
                this->tensorFlowInferencerPtr = TensorFlowInferencerPtr( new TensorFlowInferencer(netConfig, netWeights,classNames));
                break;

            case INF_KERAS:
                this->kerasInferencerPtr = KerasInferencerPtr( new KerasInferencer(netConfig, netWeights,classNames));
                break;
#ifdef ENABLE_DNN_CAFFE
            case INF_CAFFE:
                this->caffeInferencerPtr = CaffeInferencerPtr (new CaffeInferencer(netConfig, netWeights, classNames, inferencerParamsMap));
                break;
#endif
          // If it does not belong to any of the 4 supported inferencers, log warning and break.
            default:
                LOG(WARNING)<<implementation + " is not a valid inferencer implementation";
                break;
        }
    }
    else{
        LOG(WARNING)<<implementation + " is not a valid inferencer implementation";
    }

}

/*
    Check if a certain inferencer is available, if available store that in data, else skip.
*/
void GenericInferencer::configureAvailablesImplementations(std::vector<std::string>& data) {
  // If darknet exists push "yolo"
#ifdef DARKNET_ACTIVE
    data.push_back("yolo");
#endif

    // Push tensorflow and keras, as they are neccessary dependencies to use this tool.
    // If they don't exist an error should have popped up while building the tool.
    data.push_back("tensorflow");
    data.push_back("keras");

// If Caffe exists push "caffe"
#ifdef ENABLE_DNN_CAFFE
    data.push_back("caffe");
#endif
}

/*
    Returns the INFERENCER_IMPLEMENTATIONS by comparing the inferencer string
    with different available implementations.
*/
INFERENCER_IMPLEMENTATIONS GenericInferencer::getImplementation(const std::string &inferencerImplementation) {
    // Check is the selected inferencer is yolo, if it matches exactly return YOLO_INF.
    if (inferencerImplementation.compare("yolo")==0){
        return INF_YOLO;
    }
    // Check is the selected inferencer is tensorflow, if it matches exactly return INF_TENSORFLOW.
    if (inferencerImplementation.compare("tensorflow")==0){
        return INF_TENSORFLOW;
    }
    // Check is the selected inferencer is keras, if it matches exactly return INF_KERAS.
    if (inferencerImplementation.compare("keras")==0){
        return INF_KERAS;
    }
    // Check is the selected inferencer is caffe, if it matches exactly return INF_CAFFE.
    if (inferencerImplementation.compare("caffe")==0){
        return INF_CAFFE;
    }
}

/*
    Return's the inferencer pointer using the data obtained from "getImplementation".
*/
FrameworkInferencerPtr GenericInferencer::getInferencer() {
    switch (imp) {
// If darknet is selected, return darknet pointer.
#ifdef DARKNET_ACTIVE
        case INF_YOLO:
            return this->darknetInferencerPtr;
#endif
//Similarly for tensorflow and keras.
        case INF_TENSORFLOW:
            return this->tensorFlowInferencerPtr;
        case INF_KERAS:
            return this->kerasInferencerPtr;
// If caffe is selected, return caffe pointer.
#ifdef ENABLE_DNN_CAFFE
        case INF_CAFFE:
            return this->caffeInferencerPtr;
#endif

        default:
            LOG(WARNING)<<imp + " is not a valid reader implementation";
            break;
    }
}

/*
    Get all the available implementations(libraries) and store them in data.
*/
std::vector<std::string> GenericInferencer::getAvailableImplementations() {
    std::vector<std::string> data;
    configureAvailablesImplementations(data);
    return data;
}
