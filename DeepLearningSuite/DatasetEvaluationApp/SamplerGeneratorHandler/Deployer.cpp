//
// Created by frivas on 27/03/17.
//

#include <DatasetConverters/liveReaders/GenericLiveReader.h>
#include <gui/Utils.h>
#include <FrameworkEvaluator/GenericInferencer.h>
#include <FrameworkEvaluator/MassInferencer.h>
#include "Deployer.h"
#include "SamplerGenerationHandler.h"

void
SampleGeneratorHandler::Deployer::process(QListView *deployImpList, QListView *weightsList, QListView *netConfigList,
                                          QListView *inferencerImpList, QListView *inferencerNamesList,
                                          QPushButton* stopButton, QGroupBox* deployer_params, QGroupBox* inferencer_params, const std::string &weightsPath, const std::string &cfgPath,
                                          const std::string &inferencerNamesPath, const std::string &inputInfo) {

    GenericLiveReaderPtr reader;

    try {

        reader = SamplerGenerationHandler::createLiveReaderPtr( inferencerNamesList,
                                                                                 deployImpList, deployer_params, inputInfo,inferencerNamesPath);

     } catch(const std::invalid_argument& ex) {
         LOG(WARNING)<< "Error Creating Generic Live Reader\nError Message: " << ex.what();
         return;

     }

    std::vector<std::string> weights;
    if (! Utils::getListViewContent(weightsList,weights,weightsPath+ "/")){
        LOG(WARNING)<<"Select the weightsList";
        return;
    }

    std::vector<std::string> netConfiguration;
    if (! Utils::getListViewContent(netConfigList,netConfiguration,cfgPath+ "/")){
        LOG(WARNING)<<"Select the netConfiguration";
        return;
    }

    std::vector<std::string> inferencerImp;
    if (! Utils::getListViewContent(inferencerImpList,inferencerImp,"")){
        LOG(WARNING)<<"Select the inferencer type";
        return;
    }

    std::vector<std::string> inferencerNames;
    if (! Utils::getListViewContent(inferencerNamesList,inferencerNames,inferencerNamesPath + "/")){
        LOG(WARNING)<<"Select the inferencer type";
        return;
    }

    std::map<std::string, std::string>* inferencerParamsMap = new std::map<std::string, std::string>();
    try {

        if(! Utils::getDeployerParamsContent(inferencer_params, *inferencerParamsMap)) {
            inferencerParamsMap = NULL;
        }

    } catch(std::exception& ex) {
        LOG(WARNING)<< ex.what();
        return;
    }

    GenericInferencerPtr inferencer(new GenericInferencer(netConfiguration[0],weights[0],inferencerNames[0],inferencerImp[0], inferencerParamsMap));
    MassInferencer massInferencer(reader->getReader(),inferencer->getInferencer(),"./tmp", true);
    massInferencer.process(false);
}
