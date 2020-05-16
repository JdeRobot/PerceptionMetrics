#include <glog/logging.h>
#include "SamplesReader.h"



SamplesReader::SamplesReader(std::vector<Sample> & samples, std::string &classNamesFile) {
    this->samples = samples;
    this->classNamesFile = classNamesFile;
}
