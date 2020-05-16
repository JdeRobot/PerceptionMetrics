//
// Created by frivas on 19/02/17.
//

#include <glog/logging.h>
#include <gui/Utils.h>
#include "SamplerGenerationHandler.h"

GenericDatasetReaderPtr SampleGeneratorHandler::SamplerGenerationHandler::createDatasetReaderPtr(
        const QListView *datasetList,
        const QListView *namesList,
        const QListView *readerImpList,
        const QListView *filterClasses,
        const std::string &datasetPath, const std::string &namesPath, const bool imagesRequired) {
    std::vector<std::string> datasetsToShow;

    if (! Utils::getListViewContent(datasetList,datasetsToShow,datasetPath + "/")){
        LOG(WARNING)<<"Select at least one dataset to read";
        return GenericDatasetReaderPtr();
    }

    std::vector<std::string> names;
    if (! Utils::getListViewContent(namesList,names,namesPath+"/")){
        LOG(WARNING)<<"Select the dataset names related to the input dataset";
        return GenericDatasetReaderPtr();
    }

    std::vector<std::string> readerImplementation;
    if (! Utils::getListViewContent(readerImpList,readerImplementation,"")){
        LOG(WARNING)<<"Select the reader implementation";
        return GenericDatasetReaderPtr();
    }

    std::vector<std::string> classesToFilter;
    if (filterClasses)
        Utils::getListViewContent(filterClasses,classesToFilter,"");


    GenericDatasetReaderPtr reader;
    if (datasetsToShow.size()>1) {
        reader = GenericDatasetReaderPtr(
                new GenericDatasetReader(datasetsToShow,names[0], readerImplementation[0], imagesRequired));
    }
    else {
        reader = GenericDatasetReaderPtr(
                new GenericDatasetReader(datasetsToShow[0],names[0], readerImplementation[0], imagesRequired));
    }


    if (classesToFilter.size()){
        reader->getReader()->filterSamplesByID(classesToFilter);
    }

    return reader;
}

GenericLiveReaderPtr SampleGeneratorHandler::SamplerGenerationHandler::createLiveReaderPtr(const QListView *namesList,
                                                                                           const QListView *readerImpList,
                                                                                           const QGroupBox *deployer_params,
                                                                                           const QGroupBox *camera_params,
                                                                                           const std::string &infoPath,
                                                                                           const std::string &namesPath) {

    std::vector<std::string> names;
    if (! Utils::getListViewContent(namesList,names,namesPath+"/")){
        LOG(WARNING)<<"Select the dataset names related to the input dataset";
        return GenericLiveReaderPtr();
    }

    std::vector<std::string> readerImplementation;
    if (! Utils::getListViewContent(readerImpList,readerImplementation,"")){
        LOG(WARNING)<<"Select the reader implementation";
        return GenericLiveReaderPtr();
    }

    int cameraID;
    if(! Utils::getCameraParamsContent(camera_params, cameraID)) {
        LOG(WARNING)<<"Invalid Camera ID passed";
        return GenericLiveReaderPtr();
    }

    std::map<std::string, std::string>* deployer_params_map = new std::map<std::string, std::string>();

    try {

        if(! Utils::getDeployerParamsContent(deployer_params, *deployer_params_map)) {
            deployer_params_map = NULL;
        }

    } catch(std::exception& ex) {
        LOG(WARNING)<< ex.what();
        return GenericLiveReaderPtr();
    }

    GenericLiveReaderPtr reader;

    reader = GenericLiveReaderPtr(
            new GenericLiveReader(infoPath, names[0], readerImplementation[0],  deployer_params_map, cameraID));


    return reader;
}
