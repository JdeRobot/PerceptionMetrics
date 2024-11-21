//
// Created by frivas on 4/02/17.
//

#ifndef SAMPLERGENERATOR_SAMPLEGENERATIOAPP_H
#define SAMPLERGENERATOR_SAMPLEGENERATIOAPP_H

#include <config/config.h>
#include "pop_up.h"

class SampleGenerationApp {
public:
    SampleGenerationApp(int argc, char* argv[]);
    SampleGenerationApp(YAML::Node node);
    SampleGenerationApp(std::string filepath,bool isPath);
    ~SampleGenerationApp();
    virtual void operator()() =0;
    void process();
    Config::Properties getConfig();


protected:
    Config::Properties config;
    std::string configFilePath;
    std::vector<std::string> requiredArguments;

    bool verifyRequirements();
    int parse_arguments(const int argc, char* argv[], std::string& configFile);
    int argc;
    char** argv;
private:
    std::string *path;
};


#endif //SAMPLERGENERATOR_SAMPLEGENERATIOAPP_H
