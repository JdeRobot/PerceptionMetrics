//
// Created by frivas on 26/03/17.
//

#include "VideoReader.h"

/*
    Constructor which takes the video path as input and starts
    reading the video file if it can.
*/
VideoReader::VideoReader(const std::string &videoPath):DatasetReader(true) {
  // Start reading the video.
    this->cap =  new cv::VideoCapture(videoPath);
    this->framesCount = this->cap->get(cv::CAP_PROP_FRAME_COUNT);
    this->isVideo = true;

    // check if we succeeded
    if(!cap->isOpened())  // check if we succeeded
			 throw std::invalid_argument( "Couldn't open Video file!" );

    init=false;
}

// Destructor -> Stop reading once the program ends/ or is stopped by the user
VideoReader::~VideoReader() {
    LOG(INFO) << "Releasing Video File";
    this->cap->release();

}

// Store the information in "sample" which will be later processed.
bool VideoReader::getNextSample(Sample &sample) {

    cv::Mat image;
    int count = 0;

    // Try reading the frame from a video.
    try {
        while (!cap->read(image)) {
            this->validFrame = false;
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
        this->validFrame = true;

        // If we succeeded in capturing the image ,set the sampleID to the sample count
        // which was started from the moment we initialized video capturing.
        sample.setSampleID(std::to_string(++this->sample_count));
        //And the image to the captured frame.
        sample.setColorImage(image);
        return true;
    }
    
    // If something strange happens, log the exception detected.
    catch (const std::exception &exc)
    {
          LOG(ERROR) << "Exception Detected: " << exc.what();
          return false;
    }


}
