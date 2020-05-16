//
// Created by frivas on 22/01/17.
//

#ifndef SAMPLERGENERATOR_DATASETREADER_H
#define SAMPLERGENERATOR_DATASETREADER_H

#include <string>
#include <Common/Sample.h>
#include <boost/shared_ptr.hpp>
#include <Common/EvalMatrix.h>
#include <glog/logging.h>

class DatasetReader {
public:
    DatasetReader(const bool imagesRequired);
    virtual bool getNextSample(Sample &sample);
    virtual bool getNextSamples(std::vector<Sample> &samples, int size );
    void filterSamplesByID(std::vector<std::string> filteredIDS);
    void overWriteClasses(const std::string& from, const std::string& to);
    int getNumberOfElements();
    void resetReaderCounter();
    void decrementReaderCounter(const int decrement_by = 1);
    void incrementReaderCounter(const int increment_by = 1);
    bool getSampleBySampleID(Sample** sample, const std::string& sampleID);
    bool getSampleBySampleID(Sample** sample, const long long int sampleID);
    void printDatasetStats();
    virtual bool appendDataset(const std::string& datasetPath, const std::string& datasetPrefix="");
    void addSample(Sample sample);
    std::string getClassNamesFile();
    void SetClassNamesFile(std::string *names);
    void SetClasses(const std::string& classesFile);
    std::vector<std::string>* getClassNames();
    virtual ~DatasetReader();
    bool IsVideo();
    bool IsValidFrame();
    long long int TotalFrames();
protected:
    std::vector<Sample> samples;
    //std::string datasetPath;
    int readerCounter;
    std::string classNamesFile;
    std::vector<std::string> classNames;
    bool imagesRequired;
    bool isVideo;
    bool validFrame;
    long long int framesCount;
    unsigned int skip_count = 10;           //max Number of annotations that can be skipped if Corresponding images weren't found
};


typedef boost::shared_ptr<DatasetReader> DatasetReaderPtr;

#endif //SAMPLERGENERATOR_DATASETREADER_H
