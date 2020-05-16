//
// Created by frivas on 5/02/17.
//

#include "OwnDatasetWriter.h"

OwnDatasetWriter::OwnDatasetWriter(const std::string &outPath, DatasetReaderPtr &reader) : DatasetWriter(outPath,
                                                                                                       reader) {

}

void OwnDatasetWriter::process(bool writeImages, bool useDepth) {
    Sample sample;
    while (reader->getNextSample(sample)){
        sample.save(outPath);
    }
}
