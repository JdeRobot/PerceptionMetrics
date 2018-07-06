//
// Created by frivas on 22/01/17.
//

#include <unordered_map>
#include "DatasetReader.h"
#include "DatasetConverters/ClassTypeOwn.h"


DatasetReader::DatasetReader():readerCounter(0) {
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

bool DatasetReader::getNextSample(Sample &sample) {
    std::cout << "readCounter: " << this->readerCounter << ", size: " << this->samples.size() << std::endl;
    if (this->readerCounter < this->samples.size()){
        sample=this->samples[this->readerCounter];
        this->readerCounter++;
        return true;
    }
    else{
        return false;
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

    std::cout << "------------------------------------------" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    int totalSamples=0;
    for (auto it = classStats.begin(), end = classStats.end(); it != end; ++it){
        std::cout << "["<< it->first <<  "]: " << it->second << std::endl;
        totalSamples+=it->second;
    }
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "-- Total samples: " << totalSamples << std::endl;
    std::cout << "-- Total images: " << this->getNumberOfElements() << std::endl;
    std::cout << "------------------------------------------" << std::endl;


}

std::string DatasetReader::getClassNamesFile() {
    return this->classNamesFile;
}

void DatasetReader::addSample(Sample sample) {
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
                if (cr.id.compare(from) == 0) {
                    cr.id = to;
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
