//
// Created by frivas on 18/02/17.
//

#include <iostream>
#include <DatasetConverters/readers/GenericDatasetReader.h>
#include <Common/Sample.h>
#include "Viewer.h"
#include "SamplerGenerationHandler.h"
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <gui/Utils.h>

namespace SampleGeneratorHandler {

    void Viewer::process(QListView* datasetList,QListView* namesList,QListView* readerImpList,QListView* filterClasses, bool showDepth, const std::string& datasetPath, const std::string& namesPath) {

        GenericDatasetReaderPtr reader = SamplerGenerationHandler::createDatasetReaderPtr(datasetList, namesList,
                                                                                          readerImpList, filterClasses,
                                                                                          datasetPath, namesPath);
        if (!reader){
            return;
        }

        std::string windowName="viewer";
        Sample sample;

        std::vector<std::string> readerImplementation;
        Utils::getListViewContent(readerImpList,readerImplementation,"");


        while (reader->getReader()->getNextSample(sample)){
            std::cout << "number of elements: " << sample.getRectRegions()->getRegions().size() << std::endl;
            cv::Mat image =sample.getSampledColorImage();
            cv::imshow(windowName, image);

            if (showDepth) {
                if (!sample.isDepthImageValid()) {
                  LOG(WARNING)<< "Depth Images not available! Please verify your dataset or uncheck 'Show Depth Images'";
                  return;
                }

                cv::Mat depth_color;

                if (readerImplementation[0] == "spinello")
                    depth_color = sample.getSampledDepthColorMapImage(-0.9345, 1013.17);
                else
                    depth_color = sample.getSampledDepthColorMapImage();


                cv::imshow("Depth Color Map", depth_color);
            }

            int key = cv::waitKey(0);
            if (char(key) == 'q'){
                break;
            }
        }
        cv::destroyWindow(windowName);

    }

}
