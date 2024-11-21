//
// Created by frivas on 27/03/17.
//

#ifndef SAMPLERGENERATOR_DEPLOYER_H
#define SAMPLERGENERATOR_DEPLOYER_H

#include <QtWidgets/QListView>
#include <QPushButton>
#include <QMessageBox>

namespace SampleGeneratorHandler {
    class Deployer {
    public:
        static void  process(QListView *deployImpList,QListView* weightsList, QListView* netConfigList, QListView* inferencerImpList, QListView* inferencerNamesList,
                             bool* stopButton, double* confidence_threshold,  QGroupBox* deployer_params, QGroupBox* camera_params, QGroupBox* inferencer_params, const std::string& weightsPath, const std::string& cfgPath,
                             const std::string& inferencerNamesPath,const std::string& inputInfo,const std::string& outputFolder);
    };
}

#endif //SAMPLERGENERATOR_DEPLOYER_H
