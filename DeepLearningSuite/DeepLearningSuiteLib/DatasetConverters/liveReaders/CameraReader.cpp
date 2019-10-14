#include "CameraReader.h"

/* Constructor function which starts taking input from the specified webcam(deviceId) */
CameraReader::CameraReader(const int deviceId):DatasetReader(true) {
    LOG(INFO) << "Starting Capture from device with DeviceId: " << deviceId;

    // Start capturing (this is the standard way using OpenCV)
    this->cap =  new cv::VideoCapture(deviceId);

    // check if we succeeded
    if(!cap->isOpened())
			 throw std::invalid_argument( "Couldn't open Video file!" );

    // Don't know why is this set to false ! Need help !!
    init=false;
}

// Destructor -> Stop capturing if the program ends/ or is stopped by the user
CameraReader::~CameraReader() {
    LOG(INFO) << "Releasing Camera";
    this->cap->release();
}

// Store the information in "sample" which will be later processed.
bool CameraReader::getNextSample(Sample &sample) {

    cv::Mat image;
    int count = 0;
    try {
      // Try capturing the video frame
        while (!cap->read(image)) {
          LOG(ERROR) << "Frame not valid " << std::endl;
          // If we get an invalid frame for more than 5 times continously, we
          // assume the video has ended.
			    if (count >= 5) {
				    LOG(INFO) << "Video Ended" << '\n';
				    return false;
	         }						// Video Ended
	        count++;
        }
        //    init=true;

        // If we succeeded in capturing the image ,set the sampleID to the sample count
        // which was started from the moment we initialized video capturing.
        sample.setSampleID(std::to_string(++this->sample_count));

        //And the image to the captured frame.
        sample.setColorImage(image);
        return true;
    }

    // If something strange happens, log the exception detected.
    catch (const std::exception &exc){
          LOG(ERROR) << "Exception Detected: " << exc.what();
          return false;
    }

}
