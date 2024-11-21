//
// Created by frivas on 20/02/17.
//

#ifndef SAMPLERGENERATOR_DETECTOR_H
#define SAMPLERGENERATOR_DETECTOR_H


#include <QtWidgets/QListView>


namespace SampleGeneratorHandler {

    class Detector {
    public:
            static void process(QListView* datasetList,QListView* namesList,QListView* readerImpList, const std::string& datasetPath,
                                QListView* weightsList, QListView* netConfigList, QListView* inferencerImpList, QListView* inferencerNamesList,
                                QGroupBox* inferencer_params, const std::string& weightsPath, const std::string& cfgPath, const std::string& outputPath,
                                const std::string& namesPath, bool useDepth, bool singleEvaluation
            );
    };

}

#endif //SAMPLERGENERATOR_DETECTOR_H
