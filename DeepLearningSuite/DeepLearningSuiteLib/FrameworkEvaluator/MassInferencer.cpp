//
// Created by frivas on 1/02/17.
//

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <glog/logging.h>
#include "MassInferencer.h"

MassInferencer::MassInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, const std::string& resultsPath, bool debug)
 : MassInferencer::MassInferencer(reader, inferencer, resultsPath, NULL, debug) {}  // Delegating Constructor

MassInferencer::MassInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, bool debug)
 : MassInferencer::MassInferencer(reader, inferencer, NULL, debug) {}  // Delegating Constructor

MassInferencer::MassInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer,
                               const std::string &resultsPath,double* confidence_threshold, bool debug): reader(reader), inferencer(inferencer), resultsPath(resultsPath),confidence_threshold(confidence_threshold),debug(debug)
{
    if (resultsPath.empty())
        saveOutput = false;
    else
        saveOutput = true;

    alreadyProcessed=0;
    if (!resultsPath.empty()) {
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
}

MassInferencer::MassInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer,
                               const std::string &resultsPath, bool* stopDeployer,double* confidence_threshold, bool debug): reader(reader), inferencer(inferencer), resultsPath(resultsPath),debug(debug),stopDeployer(stopDeployer),confidence_threshold(confidence_threshold)
{

    if (resultsPath.empty())
        saveOutput = false;
    else
        saveOutput = true;

    alreadyProcessed=0;
    if (!resultsPath.empty()) {
        auto boostPath= boost::filesystem::path(this->resultsPath);
        if (!boost::filesystem::exists(boostPath)){
            boost::filesystem::create_directories(boostPath);
        }
        else{
            LOG(WARNING)<<"Output directory already exists";
            LOG(WARNING)<<"Files might be overwritten, if present in the directory";
            boost::filesystem::directory_iterator end_itr;


        }

    }

}

MassInferencer::MassInferencer(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, double* confidence_threshold, bool debug): reader(reader), inferencer(inferencer), confidence_threshold(confidence_threshold), debug(debug)
{
        //Constructor to avoid writing results to outputPath
        saveOutput = false;
        alreadyProcessed=0;
}

void MassInferencer::process(bool useDepthImages, DatasetReaderPtr readerDetection) {

    Sample sample;
    int counter=0;
    int nsamples = this->reader->getNumberOfElements();
    while (alreadyProcessed>0){
        LOG(INFO) << "Already evaluated: " << sample.getSampleID() << "(" << counter << "/" << nsamples << ")" << std::endl;
        this->reader->getNextSample(sample);
        counter++;
        alreadyProcessed--;
    }


    while (this->reader->getNextSample(sample)){
        counter++;
        if (this->stopDeployer != NULL && *(this->stopDeployer)) {
            LOG(INFO) << "Deployer Process Stopped" << "\n";
            return;
        }

        LOG(INFO) << "Evaluating : " << sample.getSampleID() << "(" << counter << "/" << nsamples << ")" << std::endl;

        cv::Mat image2detect;
        if (useDepthImages)
            image2detect = sample.getDepthColorMapImage();
        else {
            image2detect = sample.getColorImage();
        }

        Sample detection;

        double thresh = this->confidence_threshold == NULL ? this->default_confidence_threshold
                                                            : *(this->confidence_threshold);

        try {

          detection=this->inferencer->detect(image2detect, thresh);

        } catch(const std::runtime_error& error) {
          LOG(ERROR) << "Error Occured: " << error.what() << '\n';
          exit(1);
        }

        detection.setSampleID(sample.getSampleID());

        if (saveOutput)
            detection.save(this->resultsPath);

        if (this->debug) {
            cv::Mat image =sample.getSampledColorImage();
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
            cv::waitKey(100);
        }

        detection.clearColorImage();
        detection.clearDepthImage();

        if (readerDetection != NULL) {
            readerDetection->addSample(detection);
            //samples->push_back(detection);
        }

    }
    cv::destroyAllWindows();
    LOG(INFO) << "Mean inference time: " << this->inferencer->getMeanDurationTime() << "(ms)" <<  std::endl;


}

FrameworkInferencerPtr MassInferencer::getInferencer() const {
    return this->inferencer;
}
