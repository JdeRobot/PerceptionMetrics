//
// Created by frivas on 18/02/17.
//

#include <iostream>
#include <DatasetConverters/readers/GenericDatasetReader.h>
#include <Common/Sample.h>
#include "Viewer.h"
#include "SamplerGenerationHandler.h"
#include <glog/logging.h>
#include <gui/Utils.h>

namespace SampleGeneratorHandler {

    void Viewer::process(QListView* datasetList,QListView* namesList,QListView* readerImpList,QListView* filterClasses, bool showDepth, const std::string& datasetPath, const std::string& namesPath) {

        GenericDatasetReaderPtr reader = SamplerGenerationHandler::createDatasetReaderPtr(datasetList, namesList,
                                                                                          readerImpList, filterClasses,
                                                                                          datasetPath, namesPath, true);
        if (!reader){
            return;
        }

        std::string windowName="viewer";
        Sample sample;

        std::vector<std::string> readerImplementation;
        Utils::getListViewContent(readerImpList,readerImplementation,"");


        while (reader->getReader()->getNextSample(sample)){
            LOG(INFO) << "number of elements: " << sample.getRectRegions()->getRegions().size() << std::endl;

            if (!sample.show(readerImplementation[0], windowName, 0, showDepth))
                break;

        }
        //cv::destroyWindow(windowName);

    }

}
