//
// Created by frivas on 4/02/17.
//

#ifndef SAMPLERGENERATOR_LIVEREADER_H
#define SAMPLERGENERATOR_LIVEREADER_H

#include <string>
#include <DatasetConverters/readers/DatasetReader.h>
#include <DatasetConverters/liveReaders/VideoReader.h>
#include <DatasetConverters/liveReaders/CameraReader.h>
#include <DatasetConverters/liveReaders/JderobotReader.h>

enum LIVEREADER_IMPLEMENTATIONS{RECORDER, STREAM, CAMERA, VIDEO};

/*
  A generic reader(one for all kind of) which has all kinds of reader
  datatypes and implementations in it.
*/
class GenericLiveReader {
public:
    GenericLiveReader(const std::string& path, const std::string& classNamesFile, const std::string& readerImplementation, std::map<std::string, std::string>* deployer_params_map = NULL, int cameraID = -1);
    GenericLiveReader(const std::vector<std::string>& paths,const std::string& classNamesFile, const std::string& readerImplementation);

    DatasetReaderPtr getReader();

    static std::vector<std::string> getAvailableImplementations();

private:

  // One datatype each, for different kinds of readers.
    LIVEREADER_IMPLEMENTATIONS imp;
    VideoReaderPtr videoReaderPtr;
    CameraReaderPtr cameraReaderPtr;
    JderobotReaderPtr jderobotReaderPtr;

    std::vector<std::string> availableImplementations;

    static void configureAvailablesImplementations(std::vector<std::string>& data);
    LIVEREADER_IMPLEMENTATIONS getImplementation(const std::string& readerImplementation);
};


typedef boost::shared_ptr<GenericLiveReader> GenericLiveReaderPtr;

#endif //SAMPLERGENERATOR_GENERICDATASETREADER_H
