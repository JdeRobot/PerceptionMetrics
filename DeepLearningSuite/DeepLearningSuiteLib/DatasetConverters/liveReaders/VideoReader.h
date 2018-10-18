//
// Created by frivas on 26/03/17.
//

#ifndef SAMPLERGENERATOR_VIDEOREADER_H
#define SAMPLERGENERATOR_VIDEOREADER_H



#include <DatasetConverters/readers/DatasetReader.h>
#include <glog/logging.h>

class VideoReader: public DatasetReader {
public:
    VideoReader(const std::string& videoPath);
    ~VideoReader();

    bool getNextSample(Sample &sample);
private:
    cv::VideoCapture* cap;
    bool init;
    long long int sample_count = 0;
};

typedef boost::shared_ptr<VideoReader> VideoReaderPtr;




#endif //SAMPLERGENERATOR_VIDEOREADER_H
