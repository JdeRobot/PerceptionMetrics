//
// Created by frivas on 1/02/17.
//

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <glog/logging.h>
#include "labelling.h"

Labelling::Labelling(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, const std::string& resultsPath, bool debug)
 : Labelling::Labelling(reader, inferencer, resultsPath, NULL, debug) {
        this->detections = new std::vector<Sample>();
        this->Drawing = false;
        this->Adjusting = false;
 }  // Delegating Constructor

Labelling::Labelling(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, bool debug)
 : Labelling::Labelling(reader, inferencer, NULL, debug) {
       this->detections = new std::vector<Sample>();
       this->Drawing = false;
       this->Adjusting = false;
 }  // Delegating Constructor

Labelling::Labelling(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer,
                               const std::string &resultsPath,double* confidence_threshold, bool debug): reader(reader), inferencer(inferencer), resultsPath(resultsPath),confidence_threshold(confidence_threshold),debug(debug)
{

    if (resultsPath.empty())
        saveOutput = false;
    else
        saveOutput = true;
    alreadyProcessed=0;
    this->detections = new std::vector<Sample>();
    this->Drawing = false;
    this->Adjusting = false;
    int time=0;
    time = reader->IsVideo() ? reader->TotalFrames() : 1 ;
    this->playback.AddTrackbar(time);
    LOG(INFO) << reader->getClassNamesFile() << std::endl;
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

Labelling::Labelling(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer,
                               const std::string &resultsPath, bool* stopDeployer,double* confidence_threshold, bool debug): reader(reader), inferencer(inferencer), resultsPath(resultsPath),debug(debug),stopDeployer(stopDeployer),confidence_threshold(confidence_threshold)
{

    if (resultsPath.empty())
        saveOutput = false;
    else
        saveOutput = true;

    this->detections = new std::vector<Sample>();
    this->Drawing = false;
    this->Adjusting = false;
    LOG(INFO) << reader->getClassNamesFile() << std::endl;
    int time=0;
    time = reader->IsVideo() ? reader->TotalFrames() : 1 ;
    this->playback.AddTrackbar(time);
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

Labelling::Labelling(DatasetReaderPtr reader, FrameworkInferencerPtr inferencer, double* confidence_threshold, bool debug): reader(reader), inferencer(inferencer), confidence_threshold(confidence_threshold), debug(debug)
{
        //Constructor to avoid writing results to outputPath
        saveOutput = false;
        alreadyProcessed=0;
        LOG(INFO) << reader->getClassNamesFile() << std::endl;
        this->detections = new std::vector<Sample>();
        this->Drawing = false;
        this->Adjusting = false;
        int time=0;
        time = reader->IsVideo() ? reader->TotalFrames() : 1 ;
        this->playback.AddTrackbar(time);
}

void DrawRectangle(cv::Mat& img, cv::Rect &box){
	cv::rectangle(img,box.tl(), box.br(),cv::Scalar(0,0,0));
  cv::imshow("Detection", img);
}


void Labelling::BorderChange(int event, int x, int y, int flags, void* userdata){
  int currFrame = ((Labelling *)(userdata))->playback.currentFrame();
  bool changed = false;
  cv::Mat imager;
  switch (event) {
    case cv::EVENT_MBUTTONDOWN:{
      (((Labelling *)(userdata))->detections)->at(currFrame-1).SetClassy(x,y,((Labelling *)(userdata))->reader->getClassNames());
      changed = true;
      imager = (((Labelling *)(userdata))->detections)->at(currFrame-1).getSampledColorImage();
    }break;

    case cv::EVENT_LBUTTONDOWN :{
      if( (((Labelling *)(userdata))->detections)->at(currFrame-1).AdjustBox(x,y)){
        changed = true;
        imager = (((Labelling *)(userdata))->detections)->at(currFrame-1).getSampledColorImage();
        ((Labelling *)(userdata))->Adjusting = true;
      }
      else{
          ((Labelling *)(userdata))->Drawing = true;
          ((Labelling *)(userdata))->g_rectangle = cv::Rect(x, y, 0, 0);
      }
    }break;

    case cv::EVENT_LBUTTONUP   :{
      if( ((Labelling *)(userdata))->Adjusting ){
        (((Labelling *)(userdata))->detections)->at(currFrame-1).AdjustBox(x,y);
        changed = true;
        imager = (((Labelling *)(userdata))->detections)->at(currFrame-1).getSampledColorImage();
        ((Labelling *)(userdata))->Adjusting = false;
      }
      else if(((Labelling *)(userdata))->Drawing){
        ((Labelling *)(userdata))->Drawing = false;
        if (((Labelling *)(userdata))->g_rectangle.width < 0) {
          ((Labelling *)(userdata))->g_rectangle.x += ((Labelling *)(userdata))->g_rectangle.width;
          ((Labelling *)(userdata))->g_rectangle.width *= -1;
        }

        if (((Labelling *)(userdata))->g_rectangle.height < 0) {
          ((Labelling *)(userdata))->g_rectangle.y += ((Labelling *)(userdata))->g_rectangle.height;
          ((Labelling *)(userdata))->g_rectangle.height *= -1;
        }

        (((Labelling *)(userdata))->detections)->at(currFrame-1).AddDetection(((Labelling *)(userdata))->g_rectangle,((Labelling *)(userdata))->reader->getClassNames());
        imager = (((Labelling *)(userdata))->detections)->at(currFrame-1).getSampledColorImage();
        changed=true;
      }
    }break;

    case cv::EVENT_MOUSEMOVE: {    // When mouse moves, get the current rectangle's width and height
      if (((Labelling *)(userdata))->Drawing) {
        ((Labelling *)(userdata))->g_rectangle.width  = x - ((Labelling *)(userdata))->g_rectangle.x;
        ((Labelling *)(userdata))->g_rectangle.height = y - ((Labelling *)(userdata))->g_rectangle.y;
        imager = (((Labelling *)(userdata))->detections)->at(currFrame-1).getSampledColorImage();
        DrawRectangle(imager, ((Labelling *)(userdata))->g_rectangle);
      }
    }break;

  }

    if(changed){
      ((Labelling *)(userdata))->playback.updateFrame(currFrame-1,&imager);
      cv::imshow("Detection", imager);
      if (((Labelling *)(userdata))->saveOutput)
          (((Labelling *)(userdata))->detections)->at(currFrame-1).save(((Labelling *)(userdata))->resultsPath);
      LOG(INFO) << "Updated\n";
    }
}

void Labelling::IsProcessed(Sample *sample, int *counter , int *nsamples){
  while (alreadyProcessed>0){
    LOG(INFO) << "Already evaluated: " << sample->getSampleID() << "(" << *counter << "/" << *nsamples << ")" << std::endl;
    this->reader->getNextSample(*sample);
    *counter++;
    alreadyProcessed--;
  }
}

void Labelling::Shower(Sample *sample, Sample *detection,cv::Mat *image2detect, bool &useDepthImages){
  if (this->debug) {
      cv::Mat image =sample->getSampledColorImage();
      Sample detectionWithImage=*detection;

      if (useDepthImages)
          detectionWithImage.setColorImage(sample->getDepthColorMapImage());
      else
          detectionWithImage.setColorImage(sample->getColorImage());

      if (useDepthImages){
          cv::imshow("GT on Depth", sample->getSampledDepthColorMapImage());
          cv::imshow("Input", *image2detect);
      }
      char keystroke=cv::waitKey(1);
      if(reader->IsValidFrame() && reader->IsVideo())
        this->playback.GetInput(keystroke,detectionWithImage.getSampledColorImage(),image);
      else{
        cv::imshow("GT on RGB", image);
        cv::imshow("Detection", detectionWithImage.getSampledColorImage());
        cv::waitKey(100);
      }
  }
}

void Labelling::finder(Sample *sample , Sample *detection, cv::Mat *image2detect ,bool &useDepthImages, int *counter , int *nsamples){
  *counter+=1;
  if (this->stopDeployer != NULL && *(this->stopDeployer)) {
      LOG(INFO) << "Deployer Process Stopped" << "\n";
      return;
  }

  LOG(INFO) << "Evaluating : " << sample->getSampleID() << "(" << *counter << "/" << *nsamples << ")" << std::endl;

  if (useDepthImages)
      *image2detect = sample->getDepthColorMapImage();
  else {
      *image2detect = sample->getColorImage();
  }

  double thresh = this->confidence_threshold == NULL ? this->default_confidence_threshold
                                                      : *(this->confidence_threshold);

  try {
      *detection=this->inferencer->detect(*image2detect, thresh);
  }
  catch(const std::runtime_error& error) {
    LOG(ERROR) << "Error Occured: " << error.what() << '\n';
    exit(1);
  }

  detection->setSampleID(sample->getSampleID());

  if (saveOutput)
      detection->save(this->resultsPath);

}

void Labelling::process(bool useDepthImages, DatasetReaderPtr readerDetection) {

    Sample sample;
    int counter=0;
    int nsamples = this->reader->getNumberOfElements();

    Labelling::IsProcessed(&sample,&counter,&nsamples);
    cv::Mat image2detect;
    static Sample detection;
    cv::setMouseCallback("Detection", Labelling::BorderChange ,this);
    bool read_succesful = true;
    while (read_succesful){
        if(!detection.GetMousy()){
          read_succesful=this->reader->getNextSample(sample);
          Labelling::finder(&sample,&detection,&image2detect,useDepthImages,&counter,&nsamples);
          Labelling::Shower(&sample,&detection,&image2detect,useDepthImages);
          this->detections->push_back(detection);
      }
        detection.clearColorImage();
        detection.clearDepthImage();
        detection.SetMousy(false);
        if (readerDetection != NULL)
            readerDetection->addSample(detection);
    }

    if(!this->reader->IsValidFrame()){
      this->playback.completeShow();
      cv::destroyAllWindows();
      LOG(INFO) << "Mean inference time: " << this->inferencer->getMeanDurationTime() << "(ms)" <<  std::endl;
    }


}



FrameworkInferencerPtr Labelling::getInferencer() const {
    return this->inferencer;
}
