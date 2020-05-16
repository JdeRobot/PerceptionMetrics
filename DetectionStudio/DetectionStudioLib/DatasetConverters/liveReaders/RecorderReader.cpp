//
// Created by frivas on 16/11/16.
//

#include "RecorderReader.h"
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <Utils/PathHelper.h>
#include <boost/algorithm/string/erase.hpp>
#include <glog/logging.h>


RecorderReader::RecorderReader(const std::string &colorImagesPath, const std::string &depthImagesPath):DatasetReader(true), colorPath(colorImagesPath), depthPath(depthImagesPath) {
    currentIndex=0;
    syncedData=false;
    getImagesByIndexes(depthPath,depthIndexes);
    getImagesByIndexes(colorPath,colorIndexes);
}


RecorderReader::RecorderReader(const std::string &dataPath):DatasetReader(true), colorPath(dataPath), depthPath(dataPath) {
    currentIndex=0;
    syncedData=true;
    getImagesByIndexes(dataPath,depthIndexes,"-depth");
    getImagesByIndexes(dataPath,colorIndexes,"-rgb");
}


void RecorderReader::getImagesByIndexes(const std::string& path, std::vector<int>& indexes,std::string sufix){
    indexes.clear();
    if(boost::filesystem::is_directory(path)) {


        boost::filesystem::directory_iterator end_iter;

        for (boost::filesystem::directory_iterator dir_itr(path);
             dir_itr != end_iter; dir_itr++) {

            if (boost::filesystem::is_regular_file(*dir_itr) && dir_itr->path().extension() == ".png") {
                std::string onlyIndexFilename;
                if (not sufix.empty()) {
                    std::string filename=dir_itr->path().stem().string();
                    if ( ! boost::algorithm::ends_with(filename, sufix)){
                        continue;
                    }
                    onlyIndexFilename=dir_itr->path().filename().stem().string();
                    boost::erase_all(onlyIndexFilename,sufix);
                }
                else{
                    onlyIndexFilename=dir_itr->path().filename().stem().string();
                }
                LOG(INFO) << dir_itr->path().string() << std::endl;
                LOG(INFO) << onlyIndexFilename << std::endl;

                indexes.push_back(std::stoi(onlyIndexFilename));
            }
        }
    }
    if (indexes.empty()){
        DLOG(WARNING) << "No images found in input sample path";
    }
    std::sort(indexes.begin(), indexes.end());
}


std::string RecorderReader::getPathByIndex(const std::string& path, int id,std::string sufix){
    std::stringstream ss;
    ss << id << sufix << ".png";
    std::string pathCompleted = PathHelper::concatPaths(path, ss.str());
    return pathCompleted;
}



int RecorderReader::closest(std::vector<int> const& vec, int value) {
    auto const it = std::lower_bound(vec.begin(), vec.end(), value);
    if (it == vec.end()) { return -1; }

    return *it;
}

bool RecorderReader::getNextSample(Sample &sample) {
    if (this->currentIndex < this->depthIndexes.size()){
        int indexValue = this->depthIndexes[currentIndex];
        LOG(INFO)<<"Time stamp: " + std::to_string(indexValue);


        cv::Mat colorImage= cv::imread(getPathByIndex(this->colorPath,closest(colorIndexes,indexValue),this->syncedData?"-rgb":""));
//        if (!this->syncedData)
            cv::cvtColor(colorImage,colorImage,cv::COLOR_RGB2BGR);

        sample.setColorImage(colorImage);
        sample.setDepthImage(getPathByIndex(this->depthPath,indexValue,this->syncedData?"-depth":""));
        this->currentIndex++;
        return true;
    }
    return false;
}

int RecorderReader::getNumSamples() {
    return (int)this->depthIndexes.size();
}
