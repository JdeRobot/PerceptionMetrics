//
// Created by frivas on 29/07/17.
//

#ifndef SAMPLERGENERATOR_PRINCETONDATASETREADER_H
#define SAMPLERGENERATOR_PRINCETONDATASETREADER_H
#include <DatasetConverters/readers/DatasetReader.h>


class PrincetonDatasetReader: public DatasetReader {
public:
    PrincetonDatasetReader(const std::string& path,const std::string& classNamesFile, const bool imagesRequired);
    PrincetonDatasetReader()= default;
    virtual bool appendDataset(const std::string& datasetPath, const std::string& datasetPrefix="");

private:
};


typedef boost::shared_ptr<PrincetonDatasetReader> PrincetonDatasetReaderPtr;



#endif //SAMPLERGENERATOR_PRINCETONDATASETREADER_H
