//
// Created by frivas on 19/02/17.
//

#ifndef SAMPLERGENERATOR_SAMPLERGENERATIONHANDLER_H
#define SAMPLERGENERATOR_SAMPLERGENERATIONHANDLER_H

#include <DatasetConverters/readers/GenericDatasetReader.h>
#include <QtWidgets/QListView>
#include <QGroupBox>
#include <DatasetConverters/liveReaders/GenericLiveReader.h>
#include <glog/logging.h>

namespace SampleGeneratorHandler {

    class SamplerGenerationHandler {
    public:
        static GenericDatasetReaderPtr createDatasetReaderPtr(const QListView *datasetList, const QListView *namesList,
                                                              const QListView *readerImpList,
                                                              const QListView *filterClasses,
                                                              const std::string &datasetPath,
                                                              const std::string &namesPath,
                                                              const bool imagesRequired);

        static GenericLiveReaderPtr createLiveReaderPtr(const QListView *namesList,
                                                        const QListView *readerImpList,
                                                        const QGroupBox *deployer_params,
                                                        const QGroupBox *camera_params,
                                                        const std::string &infoPath,
                                                        const std::string &namesPath);
    };

}

#endif //SAMPLERGENERATOR_SAMPLERGENERATIONHANDLER_H
