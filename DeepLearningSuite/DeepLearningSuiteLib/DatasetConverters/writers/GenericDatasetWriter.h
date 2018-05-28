//
// Created by frivas on 5/02/17.
//

#ifndef SAMPLERGENERATOR_GENERICDATASETWRITTER_H
#define SAMPLERGENERATOR_GENERICDATASETWRITTER_H

enum WRITER_IMPLEMENTATIONS{WR_OWN, WR_YOLO, WR_COCO};

#include <string>
#include "DatasetWriter.h"
#include <DatasetConverters/writers/OwnDatasetWriter.h>
#include "YoloDatasetWriter.h"
#include "COCODatasetWriter.h"

class GenericDatasetWriter {
public:
    GenericDatasetWriter(const std::string& path,DatasetReaderPtr &reader, const std::string& writerImplementation);
    DatasetWriterPtr getWriter();
    static std::vector<std::string> getAvailableImplementations();

private:
    WRITER_IMPLEMENTATIONS imp;


    YoloDatasetWriterPtr yoloDatasetWriterPtr;
    OwnDatasetWriterPtr ownDatasetWriterPtr;
    COCODatasetWriterPtr cocoDatasetWriterPtr;

    std::vector<std::string> availableImplementations;

    static void configureAvailableImplementations(std::vector<std::string>& data);
    WRITER_IMPLEMENTATIONS getImplementation(const std::string& writerImplementation);

};


typedef boost::shared_ptr<GenericDatasetWriter> GenericDatasetWriterPtr;


#endif //SAMPLERGENERATOR_GENERICDATASETWRITTER_H
