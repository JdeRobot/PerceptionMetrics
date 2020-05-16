//
// Created by frivas on 20/02/17.
//

#include <DatasetConverters/readers/GenericDatasetReader.h>
#include <DatasetConverters/writers/GenericDatasetWriter.h>
#include <gui/Utils.h>
#include <glog/logging.h>
#include <FrameworkEvaluator/GenericInferencer.h>
#include <FrameworkEvaluator/MassInferencer.h>
#include "Detector.h"
#include "SamplerGenerationHandler.h"

void SampleGeneratorHandler::Detector::process(QListView* datasetList,QListView* namesList,QListView* readerImpList, const std::string& datasetPath,
                                               QListView* weightsList, QListView* netConfigList, QListView* inferencerImpList, QListView* inferencerNamesList,
                                               QGroupBox* inferencer_params, const std::string& weightsPath, const std::string& cfgPath, const std::string& outputPath,
                                               const std::string& namesPath, bool useDepth, bool singleEvaluation) {

    GenericDatasetReaderPtr reader = SamplerGenerationHandler::createDatasetReaderPtr(datasetList, namesList,
                                                                                      readerImpList, NULL, datasetPath,
                                                                                      namesPath, true);

    if (!reader)
        return;
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
    if (! Utils::getListViewContent(inferencerNamesList,inferencerNames,namesPath + "/")){
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

    std::vector<std::string> writerImp;
    Utils::getListViewContent(readerImpList,writerImp,"");


    std::vector<std::string> writerNames;
    if (! Utils::getListViewContent(namesList,writerNames,namesPath+"/")){
        LOG(WARNING)<<"Select the dataset names related to the Output dataset, or unchechk mapping if you want a custom names file to be generated";
        return;
    }

    DatasetReaderPtr readerDetection ( new DatasetReader(true) );

    GenericInferencerPtr inferencer(new GenericInferencer(netConfiguration[0],weights[0],inferencerNames[0],inferencerImp[0], inferencerParamsMap));
    MassInferencer massInferencer(reader->getReader(),inferencer->getInferencer(),std::string(), true);
    massInferencer.process(useDepth, readerDetection);

    GenericDatasetWriterPtr writer( new GenericDatasetWriter(outputPath,readerDetection,writerImp[0], writerNames[0]));

    writer->getWriter()->process(false);


}
