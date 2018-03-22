//
// Created by frivas on 4/02/17.
//

#include <glog/logging.h>
#include "GenericInferencer.h"

GenericInferencer::GenericInferencer(const std::string &netConfig, const std::string &netWeights,const std::string& classNames,
                                     const std::string &implementation) {

    configureAvailablesImplementations(this->availableImplementations);
    if (std::find(this->availableImplementations.begin(), this->availableImplementations.end(), implementation) != this->availableImplementations.end()){
        imp = getImplementation(implementation);
        switch (imp) {
#ifdef DARKNET_ACTIVE
            case INF_YOLO:
                this->darknetInferencerPtr = DarknetInferencerPtr( new DarknetInferencer(netConfig, netWeights,classNames));
                break;
#endif
            case INF_TENSORFLOW:
              this->tensorFlowInferencerPtr = TensorFlowInferencerPtr( new TensorFlowInferencer(netConfig, netWeights,classNames));
              break;

            default:
                LOG(WARNING)<<implementation + " is not a valid inferencer implementation";
                break;
        }
    }
    else{
        LOG(WARNING)<<implementation + " is not a valid inferencer implementation";
    }

}

void GenericInferencer::configureAvailablesImplementations(std::vector<std::string>& data) {
#ifdef DARKNET_ACTIVE
    data.push_back("yolo");
#endif
    data.push_back("tensorflow");
}

INFERENCER_IMPLEMENTATIONS GenericInferencer::getImplementation(const std::string &inferencerImplementation) {
    if (inferencerImplementation.compare("yolo")==0){
        return INF_YOLO;
    }
    if (inferencerImplementation.compare("tensorflow")==0){
        std::cout << "returned tensorflow" << '\n';
        return INF_TENSORFLOW;
    }
}

FrameworkInferencerPtr GenericInferencer::getInferencer() {
    switch (imp) {
#ifdef DARKNET_ACTIVE
        case INF_YOLO:
            return this->darknetInferencerPtr;
#endif
        case INF_TENSORFLOW:
            return this->tensorFlowInferencerPtr;
        default:
            LOG(WARNING)<<imp + " is not a valid reader implementation";
            break;
    }
}

std::vector<std::string> GenericInferencer::getAvailableImplementations() {
    std::vector<std::string> data;
    configureAvailablesImplementations(data);
    return data;


}
