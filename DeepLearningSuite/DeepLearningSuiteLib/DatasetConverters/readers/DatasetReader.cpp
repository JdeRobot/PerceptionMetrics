//
// Created by frivas on 22/01/17.
//

#include <unordered_map>
#include "DatasetReader.h"
#include "DatasetConverters/ClassTypeOwn.h"
#include <glog/logging.h>

DatasetReader::DatasetReader(const bool imagesRequired):readerCounter(0),imagesRequired(imagesRequired),isVideo(false),validFrame(true){
}

DatasetReader::~DatasetReader() {
}

void DatasetReader::filterSamplesByID(std::vector<std::string> filteredIDS) {
    std::vector<Sample> old_samples(this->samples);
    this->samples.clear();

    for (auto it=old_samples.begin(), end=old_samples.end(); it != end; ++it){
        Sample& sample =*it;
        sample.filterSamplesByID(filteredIDS);
        this->samples.push_back(sample);
//        if (sample.getContourRegions() && sample.getContourRegions()->empty() && sample.getRectRegions()->empty()){
//
//        }
//        else{
//        }
    }
}

int DatasetReader::getNumberOfElements() {
    return this->samples.size();
}

void DatasetReader::resetReaderCounter() {
    this->readerCounter=0;

}

void DatasetReader::decrementReaderCounter(const int decrement_by) {
    this->readerCounter -= decrement_by;
}

void DatasetReader::incrementReaderCounter(const int increment_by) {
    this->readerCounter +=  increment_by;
}

bool DatasetReader::getNextSample(Sample &sample) {
    LOG(INFO) << "readCounter: " << this->readerCounter << ", size: " << this->samples.size();
    if (this->readerCounter < this->samples.size()){
        sample=this->samples[this->readerCounter];
        this->readerCounter++;
        return true;
    }
    else{
        return false;
    }

}

bool DatasetReader::getNextSamples(std::vector<Sample> &batchOfSamples, int size ) {
    LOG(INFO) << "readCounter: " << this->readerCounter << ", size: " << this->samples.size();

    int imagesLeft = this->samples.size() - this->readerCounter;

    if ( size > imagesLeft )
        size = imagesLeft;

    if (batchOfSamples.capacity() != size)
        batchOfSamples.resize(size);


    for (int i = 0; i < size; i++) {
        batchOfSamples[i] = this->samples[this->readerCounter];
        this->readerCounter++;
    }

}

bool DatasetReader::getSampleBySampleID(Sample** sample, const std::string& sampleID){
    for (auto it=this->samples.begin(), end= this->samples.end(); it != end; ++it){
        if (it->getSampleID().compare(sampleID)==0){
            *sample=&(*it);
            return true;
        }
    }
    return false;
}

bool DatasetReader::getSampleBySampleID(Sample** sample, const long long int sampleID) {
    for (auto it=this->samples.begin(), end= this->samples.end(); it != end; ++it){
        if ((long long int)std::stoi(it->getSampleID())==sampleID){
            *sample=&(*it);
            return true;
        }
    }
    return false;
}

void DatasetReader::printDatasetStats() {
    std::unordered_map<std::string, int> classStats;
    std::unordered_map<std::string, int>::iterator map_it;

    for (auto it=samples.begin(), end=samples.end(); it != end; ++it){
        RectRegionsPtr regions = it->getRectRegions();
        if (regions) {
            std::vector<RectRegion> regionsVector = regions->getRegions();
            for (std::vector<RectRegion>::iterator itRegion = regionsVector.begin(), endRegion = regionsVector.end();
                 itRegion != endRegion; ++itRegion) {
                std::string test = itRegion->classID;

                //ClassTypeOwn typeconv(test);
                map_it = classStats.find(test);
                if (map_it != classStats.end()) {
                    map_it->second++;
                } else {
                    classStats.insert(std::make_pair(test, 1));
                }
            }
        }
    }

    LOG(INFO) << "------------------------------------------" << std::endl;
    LOG(INFO) << "------------------------------------------" << std::endl;
    int totalSamples=0;
    for (auto it = classStats.begin(), end = classStats.end(); it != end; ++it){
        LOG(INFO) << "["<< it->first <<  "]: " << it->second << std::endl;
        totalSamples+=it->second;
    }
    LOG(INFO) << "------------------------------------------" << std::endl;
    LOG(INFO) << "-- Total samples: " << totalSamples << std::endl;
    LOG(INFO) << "-- Total images: " << this->getNumberOfElements() << std::endl;
    LOG(INFO) << "------------------------------------------" << std::endl;


}

std::string DatasetReader::getClassNamesFile() {
    return this->classNamesFile;
}

void DatasetReader::addSample(Sample sample) {
    if (imagesRequired && (!sample.getColorImage().empty() || !sample.getDepthImage().empty())) {
        throw std::invalid_argument("Dataset Reader Requires Images, and sample doesn't contain any !!\n"
        "The Class which has instantiated dataset Reader, requires it to contain images, whereas the sample doesn't contain any!");
    }

    this->samples.push_back(sample);
}

bool DatasetReader::appendDataset(const std::string &datasetPath, const std::string &datasetPrefix) {
    return false;
}

void DatasetReader::overWriteClasses(const std::string &from, const std::string &to) {
    for (auto it = samples.begin(), end= samples.end(); it != end; ++it){
        Sample& s= *it;

        if (s.getContourRegions()) {
            for (auto it2 = s.getContourRegions()->regions.begin(), end2 = s.getContourRegions()->regions.end();
                 it2 != end2; ++it2) {
                ContourRegion &cr = *it2;
                if (cr.classID.compare(from) == 0) {
                    cr.classID = to;
                }
            }
        }
        for (auto it2 = s.getRectRegions()->regions.begin(), end2 = s.getRectRegions()->regions.end(); it2 != end2; ++it2){
            RectRegion& r = *it2;
            if (r.classID.compare(from)==0){
                r.classID=to;
            }
        }

    }
}

bool DatasetReader::IsValidFrame(){
  return this->validFrame;
}

bool DatasetReader::IsVideo(){
  return this->isVideo;
}

long long int DatasetReader::TotalFrames(){
  return this->framesCount;
}

void DatasetReader::SetClassNamesFile(std::string *names){
  this->classNamesFile = *names;
  DatasetReader::SetClasses(this->classNamesFile);
}

void DatasetReader::SetClasses(const std::string& classesFile){
  std::ifstream labelFile(classesFile);
  std::string data;
  this->classNames.clear();
  while(getline(labelFile,data)) {
      this->classNames.push_back(data);
  }
}

std::vector<std::string>* DatasetReader::getClassNames(){
  return &this->classNames;
}
