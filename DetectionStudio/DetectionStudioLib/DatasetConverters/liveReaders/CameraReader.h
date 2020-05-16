#ifndef SAMPLERGENERATOR_CAMERAREADER_H
#define SAMPLERGENERATOR_CAMERAREADER_H


#include <DatasetConverters/readers/DatasetReader.h>
#include <glog/logging.h>

class CameraReader: public DatasetReader {
public:
    // Constructor
    CameraReader(const int deviceId = -1);

    // Destructor
    ~CameraReader();

    //This sample address will be passed by some process like evaluator,detector,etc.
    // Later on this "sample" will be processed.
    bool getNextSample(Sample &sample);
private:
    // Pointer which stores the address of the video being captured.
    cv::VideoCapture* cap;
    bool init;

    // Counter which will be initialized from the moment we start capturing video.
    long long int sample_count = 0;

};

typedef boost::shared_ptr<CameraReader> CameraReaderPtr;




#endif //SAMPLERGENERATOR_CAMERAREADER_H
