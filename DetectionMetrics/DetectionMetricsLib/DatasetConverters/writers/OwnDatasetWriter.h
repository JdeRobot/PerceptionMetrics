//
// Created by frivas on 5/02/17.
//

#ifndef SAMPLERGENERATOR_OWNDATASETWRITER_H
#define SAMPLERGENERATOR_OWNDATASETWRITER_H

#include <string>
#include <DatasetConverters/readers/DatasetReader.h>
#include "DatasetWriter.h"

class OwnDatasetWriter: public DatasetWriter {
public:
    OwnDatasetWriter(const std::string &outPath, DatasetReaderPtr &reader);
    void process(bool writeImages = false, bool useDepth = false);

private:

};

typedef  boost::shared_ptr<OwnDatasetWriter> OwnDatasetWriterPtr;



#endif //SAMPLERGENERATOR_OWNDATASETWRITER_H
