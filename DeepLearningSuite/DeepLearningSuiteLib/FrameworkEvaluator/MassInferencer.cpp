//
// Created by frivas on 1/02/17.
//

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <glog/logging.h>
#include "MassInferencer.h"

MassInferencer::MassInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer,
                               const std::string &resultsPath,bool debug): reader(reader), inferencer(inferencer), resultsPath(resultsPath),debug(debug)
{
    saveOutput = true;
    alreadyProcessed=0;
    auto boostPath= boost::filesystem::path(this->resultsPath);
    if (!boost::filesystem::exists(boostPath)){
        boost::filesystem::create_directories(boostPath);
    }
    else{
        LOG(WARNING)<<"Output directory already exists";
        LOG(WARNING)<<"Continuing detecting";
        boost::filesystem::directory_iterator end_itr;

        for (boost::filesystem::directory_iterator itr(boostPath); itr!=end_itr; ++itr)
        {
            if ((is_regular_file(itr->status()) && itr->path().extension()==".png") && (itr->path().string().find("-depth") == std::string::npos)) {
                alreadyProcessed++;
            }

        }
        //exit(-1);
    }
}

MassInferencer::MassInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, bool debug): reader(reader), inferencer(inferencer), debug(debug)
{
        //Constructor to avoid writing results to outputPath
        saveOutput = false;
        alreadyProcessed=0;
}

void MassInferencer::process(bool useDepthImages, std::vector<Sample>* samples) {

    Sample sample;
    int counter=0;
    int nsamples = this->reader->getNumberOfElements();
    while (alreadyProcessed>0){
        std::cout << "Already evaluated: " << sample.getSampleID() << "(" << counter << "/" << nsamples << ")" << std::endl;
        this->reader->getNextSample(sample);
        counter++;
        alreadyProcessed--;
    }


    std::cout << "here" << '\n';

    while (this->reader->getNextSample(sample)){
        counter++;
        std::cout << "Evaluating : " << sample.getSampleID() << "(" << counter << "/" << nsamples << ")" << std::endl;

        cv::Mat image =sample.getSampledColorImage();
        cv::Mat image2detect;
        if (useDepthImages)
            image2detect = sample.getDepthColorMapImage();
        else {
            image2detect = sample.getColorImage();
        }

        Sample detection;

        try {

          detection=this->inferencer->detect(image2detect);

        } catch(const std::runtime_error& error) {
          std::cout << "Error Occured: " << error.what() << '\n';
          exit(1);
        }

        detection.setSampleID(sample.getSampleID());

        if (saveOutput)
            detection.save(this->resultsPath);

        if (samples != NULL) {
            samples->push_back(detection);
        }
        if (this->debug) {
            Sample detectionWithImage;
            detectionWithImage=detection;
            if (useDepthImages)
                detectionWithImage.setColorImage(sample.getDepthColorMapImage());
            else
                detectionWithImage.setColorImage(sample.getColorImage());
            cv::imshow("GT on RGB", image);
            if (useDepthImages){
                cv::imshow("GT on Depth", sample.getSampledDepthColorMapImage());
                cv::imshow("Input", image2detect);
            }
            cv::imshow("Detection", detectionWithImage.getSampledColorImage());
            cv::waitKey(10);
        }
    }
    cv::destroyAllWindows();
    std::cout << "Mean inference time: " << this->inferencer->getMeanDurationTime() << "(ms)" <<  std::endl;


}
