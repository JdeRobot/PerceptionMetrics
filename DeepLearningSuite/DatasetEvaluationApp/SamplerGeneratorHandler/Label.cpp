//
// Created by frivas on 27/03/17.
//

#include <DatasetConverters/liveReaders/GenericLiveReader.h>
#include <gui/Utils.h>
#include <FrameworkEvaluator/GenericInferencer.h>
#include <FrameworkEvaluator/MassInferencer.h>
#include "Deployer.h"
#include "SamplerGenerationHandler.h"
#include "gui/Appcfg.hpp"

void
SampleGeneratorHandler::Deployer::process(QListView *deployImpList, QListView *weightsList, QListView *netConfigList,
                                          QListView *inferencerImpList, QListView *inferencerNamesList,
                                          bool* stopButton, double* confidence_threshold, QGroupBox* deployer_params, QGroupBox* camera_params, QGroupBox* inferencer_params, const std::string &weightsPath, const std::string &cfgPath,
                                          const std::string &inferencerNamesPath, const std::string &inputInfo, const std::string &outputFolder) {

    GenericLiveReaderPtr reader;
    // LOG(INFO) << "Osat : " << inferencerNamesPath << std::endl;

    try {

        reader = SamplerGenerationHandler::createLiveReaderPtr( inferencerNamesList,
                                                                                 deployImpList, deployer_params, camera_params, inputInfo,inferencerNamesPath);

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
        if(! Utils::getInferencerParamsContent(inferencer_params, *inferencerParamsMap)) {
            inferencerParamsMap = NULL;
        }

    } catch(std::exception& ex) {
        LOG(WARNING)<< ex.what();
        return;
    }

    if (!outputFolder.empty()) {

        auto boostPath= boost::filesystem::path(outputFolder);
        if (boost::filesystem::exists(boostPath)){
            boost::filesystem::directory_iterator end_itr;
            boost::filesystem::directory_iterator itr(boostPath);
            for (; itr != end_itr; ++itr)
            {
        // If it's not a directory, list it. If you want to list directories too, just remove this check.
                if (boost::filesystem::is_regular_file(itr->path()) && (itr->path().extension()==".png" || itr->path().extension()==".json")  ) {
            // assign current file name to current_file and echo it out to the console.
                    break;
                }
            }
            if (itr != end_itr)
                QMessageBox::warning(deployer_params, QObject::tr("Output Directory isn't Empty"), QObject::tr("Output Director contains png or json files which might be overwritten"));
        }

    }
    DatasetReaderPtr data_reader=reader->getReader();
    data_reader->SetClassNamesFile(&inferencerNames[0]);
    GenericInferencerPtr inferencer(new GenericInferencer(netConfiguration[0],weights[0],inferencerNames[0],inferencerImp[0], inferencerParamsMap));
    MassInferencer massInferencer(data_reader,inferencer->getInferencer(),outputFolder, stopButton, confidence_threshold, true);
    massInferencer.process(false);
}
