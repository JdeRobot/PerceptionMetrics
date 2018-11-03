#ifndef SAMPLERGENERATOR_CAMERAREADER_H
#define SAMPLERGENERATOR_CAMERAREADER_H


#include <DatasetConverters/readers/DatasetReader.h>
#include <glog/logging.h>

class CameraReader: public DatasetReader {
public:
    CameraReader(const int deviceId = -1);
    ~CameraReader();
    bool getNextSample(Sample &sample);
private:
    cv::VideoCapture* cap;
    bool init;
    long long int sample_count = 0;

};

typedef boost::shared_ptr<CameraReader> CameraReaderPtr;




#endif //SAMPLERGENERATOR_CAMERAREADER_H
