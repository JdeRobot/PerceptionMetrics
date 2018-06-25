#ifndef SAMPLERGENERATOR_CAMERAREADER_H
#define SAMPLERGENERATOR_CAMERAREADER_H


#include <DatasetConverters/readers/DatasetReader.h>


class CameraReader: public DatasetReader {
public:
    CameraReader(const int deviceId = -1);

    bool getNextSample(Sample &sample);
private:
    cv::VideoCapture* cap;
    bool init;

};

typedef boost::shared_ptr<CameraReader> CameraReaderPtr;




#endif //SAMPLERGENERATOR_CAMERAREADER_H
