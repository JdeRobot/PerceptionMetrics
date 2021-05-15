//
// Created by frivas on 22/01/17.
//

#ifndef SAMPLERGENERATOR_OWNDATASETREADER_H
#define SAMPLERGENERATOR_OWNDATASETREADER_H

#include <DatasetConverters/readers/DatasetReader.h>

class OwnDatasetReader:public DatasetReader {
public:
    OwnDatasetReader(const std::string& path,const std::string& classNamesFile, const bool imagesRequired);
    OwnDatasetReader(const bool imagesRequired);
    bool appendDataset(const std::string& datasetPath, const std::string& datasetPrefix="");
private:

};


typedef boost::shared_ptr<OwnDatasetReader> OwnDatasetReaderPtr;


#endif //SAMPLERGENERATOR_OWNDATASETREADER_H
