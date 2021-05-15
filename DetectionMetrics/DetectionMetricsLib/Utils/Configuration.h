//
// Created by frivas on 4/02/17.
//

#ifndef SAMPLERGENERATOR_CONFIGURATION_H
#define SAMPLERGENERATOR_CONFIGURATION_H

#include <string>
#include <map>
#include <boost/shared_ptr.hpp>
#include "Key.h"

class Configuration {
public:
    Configuration();
    Configuration(const std::string& configFilePath);
    void showConfig();
    Key getKey(const std::string& key);
    bool keyExists(const std::string& key);

private:
    std::string configFilePath;
    std::map<std::string, Key> config;
};

typedef  boost::shared_ptr<Configuration> ConfigurationPtr;

#endif //SAMPLERGENERATOR_CONFIGURATION_H
