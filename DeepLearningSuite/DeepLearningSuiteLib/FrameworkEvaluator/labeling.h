//
// Created by frivas on 1/02/17.
//

#ifndef SAMPLERGENERATOR_LABELING_H
#define SAMPLERGENERATOR_LABELING_H

#include <DatasetConverters/readers/DatasetReader.h>
#include <DatasetConverters/ClassTypeGeneric.h>
#include <FrameworkEvaluator/FrameworkInferencer.h>
#include "Utils/Playback.hpp"

class Labeling {
public:
    Labeling(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, const std::string& resultsPath, double* confidence_threshold = NULL, bool debug=true);
    Labeling(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, const std::string& resultsPath, bool debug=true);
    Labeling(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, const std::string &resultsPath, bool* stopDeployer, double* confidence_threshold = NULL, bool debug=true);
    Labeling(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, double* confidence_threshold = NULL, bool debug=true);
    Labeling(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, bool debug=true);
    void process(bool writeImages, DatasetReaderPtr readerDetection = NULL);
    FrameworkInferencerPtr getInferencer() const;
    static void BorderChange(int event, int x, int y, int flags, void* userdata);
    void IsProcessed(Sample *sample, int *counter , int *nsamples);
    void Shower(Sample *sample, Sample *detection,cv::Mat *image2detect, bool &useDepthImages);
    void finder(Sample *sample, Sample *detection,cv::Mat *image2detect, bool &useDepthImages, int *counter , int *nsamples);
private:
    DatasetReaderPtr reader;
    FrameworkInferencerPtr inferencer;
    std::string resultsPath;
    bool debug;
    bool saveOutput;
    int alreadyProcessed;
    bool* stopDeployer = NULL;
    double* confidence_threshold = NULL;
    double default_confidence_threshold = 0.2;
    Playback playback;
    std::vector<Sample> *detections;
    bool mousy;
    cv::Rect g_rectangle;
    bool Drawing ;
    bool Adjusting ;
};



#endif //SAMPLERGENERATOR_MASSINFERENCER_H
