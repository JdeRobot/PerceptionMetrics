//
// Created by frivas on 19/02/17.
//

#ifndef SAMPLERGENERATOR_UTILS_H
#define SAMPLERGENERATOR_UTILS_H


#include <QtWidgets/QListView>
#include <QtWidgets>
#include <QGroupBox>
#include <QRadioButton>
#include <map>
#include <iostream>

class Utils {
public:
    static bool getListViewContent(const QListView* list,std::vector<std::string>& content ,const std::string& prefix);
    static bool getDeployerParamsContent(const QGroupBox* deployer_params, std::map<std::string, std::string>& deployer_params_map);
    static bool getInferencerParamsContent(const QGroupBox* inferencer_params, std::map<std::string, std::string>& inferencer_params_map);
    static bool getCameraParamsContent(const QGroupBox* camera_params, int& cameraID);
};


#endif //SAMPLERGENERATOR_UTILS_H
