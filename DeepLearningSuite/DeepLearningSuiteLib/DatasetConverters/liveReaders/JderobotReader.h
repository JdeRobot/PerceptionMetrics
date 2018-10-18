//
// Created by frivas on 24/02/17.
//

#ifndef SAMPLERGENERATOR_JDEROBOTREADER_H
#define SAMPLERGENERATOR_JDEROBOTREADER_H


#include <DatasetConverters/readers/DatasetReader.h>
#include <config/config.h>
#include <comm/communicator.hpp>
#include <comm/cameraClient.hpp>
#include <yaml-cpp/yaml.h>
#include <glog/logging.h>

class JderobotReader: public DatasetReader {
public:
    JderobotReader(std::map<std::string, std::string>* deployer_params_map, const std::string& path);

    bool getNextSample(Sample &sample);
private:
    Comm::Communicator* jdrc;
    Comm::CameraClient* camera;
    long long int sample_count = 0;

};

typedef boost::shared_ptr<JderobotReader> JderobotReaderPtr;


#endif //SAMPLERGENERATOR_JDEROBOTREADER_H
