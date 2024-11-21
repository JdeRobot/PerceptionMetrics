// Created by frivas on 16/11/16.
//

#ifndef SAMPLERGENERATOR_RECORDERCONVERTER_H
#define SAMPLERGENERATOR_RECORDERCONVERTER_H

#include <string>
#include <vector>
#include <Common/Sample.h>
#include "DatasetConverters/readers/DatasetReader.h"


class RecorderReader: public DatasetReader {
public:
    RecorderReader(const std::string& colorImagesPath, const std::string& depthImagesPath);
    explicit RecorderReader(const std::string& dataPath);
    bool getNextSample(Sample &sample) override;
    int getNumSamples();
//    virtual bool getNextSample(Sample &sample);


private:
    const  std::string depthPath;
    const std::string colorPath;
    bool syncedData;
    int currentIndex;
    std::vector<int> depthIndexes;
    std::vector<int> colorIndexes;

    void getImagesByIndexes(const std::string& path, std::vector<int>& indexes, std::string sufix="");
    std::string getPathByIndex(const std::string& path,int id, std::string sufix="");
    int closest(std::vector<int> const& vec, int value);


        };

        typedef  boost::shared_ptr<RecorderReader> RecorderReaderPtr;

#endif //SAMPLERGENERATOR_RECORDERCONVERTER_H
