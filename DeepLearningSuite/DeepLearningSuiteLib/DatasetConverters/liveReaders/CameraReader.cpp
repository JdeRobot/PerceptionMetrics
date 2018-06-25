#include "CameraReader.h"

CameraReader::CameraReader(const int deviceId) {
    this->cap =  new cv::VideoCapture(deviceId);

    if(!cap->isOpened())  // check if we succeeded
			 throw std::invalid_argument( "Couldn't open Video file!" );

    init=false;
}

bool CameraReader::getNextSample(Sample &sample) {

    cv::Mat image;
    int count = 0;

    try {
        while (!cap->read(image)) {
            std::cout << "Frame not valid " << std::endl;
			if (count >= 5) {
				std::cout << "Video Ended" << '\n';
				return false;
			}						// Video Ended
			count++;
        }

        //    init=true;

        sample.setColorImage(image);
        return true;
    }
    catch (const std::exception &exc)
    {
          std::cout << "Exeption Detected: " << exc.what();
          return false;
    }
    catch (...){
        return false;
    }

}
