//
// Created by frivas on 22/01/17.
//

#ifndef SAMPLERGENERATOR_YOLODATASETREADER_H
#define SAMPLERGENERATOR_YOLODATASETREADER_H


#include <DatasetConverters/readers/DatasetReader.h>

class YoloDatasetReader: public DatasetReader {
public:
    YoloDatasetReader(const std::string& path,const std::string& classNamesFile, bool imagesRequired);
    YoloDatasetReader(const std::string& classNamesFile,bool imagesRequired);
    bool appendDataset(const std::string& datasetPath, const std::string& datasetPrefix="");

};

typedef boost::shared_ptr<YoloDatasetReader> YoloDatasetReaderPtr;

#endif //SAMPLERGENERATOR_YOLODATASETREADER_H
