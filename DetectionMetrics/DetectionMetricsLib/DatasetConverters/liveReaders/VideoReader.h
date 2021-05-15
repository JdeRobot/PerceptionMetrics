//
// Created by frivas on 26/03/17.
//

#ifndef SAMPLERGENERATOR_VIDEOREADER_H
#define SAMPLERGENERATOR_VIDEOREADER_H



#include <DatasetConverters/readers/DatasetReader.h>
#include <glog/logging.h>

class VideoReader: public DatasetReader {
public:
    // Constructor which takes the videoPath as input.
    VideoReader(const std::string& videoPath);

    // Destructor
    ~VideoReader();

    //This sample address will be passed by some process like evaluator,detector,etc.
    // Later on this "sample" will be processed.
    bool getNextSample(Sample &sample);
private:
  // Pointer which stores the address of the video being read.
    cv::VideoCapture* cap;
    bool init;
    
    // Counter which will be initialized from the moment we start capturing video.
    long long int sample_count = 0;
};

typedef boost::shared_ptr<VideoReader> VideoReaderPtr;




#endif //SAMPLERGENERATOR_VIDEOREADER_H
