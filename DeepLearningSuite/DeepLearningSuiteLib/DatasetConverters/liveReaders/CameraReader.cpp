#include "CameraReader.h"

CameraReader::CameraReader(const int deviceId):DatasetReader(true) {
    LOG(INFO) << "Starting Capture from device with DeviceId: " << deviceId;
    this->cap =  new cv::VideoCapture(deviceId);

    if(!cap->isOpened())  // check if we succeeded
			 throw std::invalid_argument( "Couldn't open Video file!" );

    init=false;
}

CameraReader::~CameraReader() {
    LOG(INFO) << "Releasing Camera";
    this->cap->release();
}

bool CameraReader::getNextSample(Sample &sample) {

    cv::Mat image;
    int count = 0;

    try {
        while (!cap->read(image)) {
            LOG(ERROR) << "Frame not valid " << std::endl;
			if (count >= 5) {
				LOG(INFO) << "Video Ended" << '\n';
				return false;
			}						// Video Ended
			count++;
        }

        //    init=true;
        sample.setSampleID(std::to_string(++this->sample_count));
        sample.setColorImage(image);
        return true;
    }
    catch (const std::exception &exc)
    {
          LOG(ERROR) << "Exception Detected: " << exc.what();
          return false;
    }

}
