//
// Created by frivas on 4/02/17.
//

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include "Configuration.h"
#include <glog/logging.h>
#include <boost/algorithm/string/predicate.hpp>

Configuration::Configuration(const std::string &configFilePath):configFilePath(configFilePath) {
    boost::filesystem::path boostPath(configFilePath);
    if (boost::filesystem::exists(boostPath)){
        std::string line;
        std::ifstream inFile(configFilePath);
        std::string key;
        while (getline(inFile,line)) {
            if (boost::starts_with(line, "#")){
                continue;
            }
            if (line.empty())
                continue;
            if (boost::starts_with(line, "--")){
                key = line.erase(0,2);
                if (this->config.count(key)){
                    LOG(ERROR) << "Duplicated key in configuration file: " + key;
                    exit(1);
                }
                else {
                    Key keyConfig(key);
                    this->config[key] = keyConfig;
                }
            }
            else{
                if (key.empty()){
                    LOG(WARNING) << "Error no key detected for " + line + " value";
                }
                else{
                    this->config[key].addValue(line);
                }

            }

        }

    }
    showConfig();
}

void Configuration::showConfig() {
    LOG(INFO) << "------------------------------------------------------------------" << std::endl;
    LOG(INFO) << "------------------------------------------------------------------" << std::endl;
    for (auto it = this->config.begin(), end = this->config.end(); it != end; ++it){
        LOG(INFO) << it->first << std::endl;
        int nElements= it->second.getNValues();
        for (int i=0; i < nElements; i++) {
            LOG(INFO) << "       " << it->second.getValue(i) << std::endl;
        }
    }
    LOG(INFO) << "------------------------------------------------------------------" << std::endl;
    LOG(INFO) << "------------------------------------------------------------------" << std::endl;
}

Configuration::Configuration() {

}

Key Configuration::getKey(const std::string &key) {
    if (this->config.count(key)==0) {
        LOG(ERROR) << "Key: " + key + " does not exists inside the configuration";
        exit(1);
    }
    else
        return this->config[key];
}

bool Configuration::keyExists(const std::string& key) {
    return this->config.count(key)!=0;
}
