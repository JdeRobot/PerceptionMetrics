#include "Playback.hpp"

// Callback function that is triggered when someone uses the slidebar
void Playback::onSlide(int currentPos,void *frame){
  // updateFrame the frame to currentPos of slidebar
    *(int *)(frame) = currentPos;
}

// Constructor which is called if the frames count is known before hand
Playback::Playback(long long int framesCount):framesCount(framesCount),pause(false),speed(1),frameId(0),inferences(0),groundTruth(0){
  int *frame = &this->frameId;
  cv::namedWindow("Detection",cv::WINDOW_NORMAL);
  cv::namedWindow("GT on RGB",cv::WINDOW_NORMAL);
  cv::createTrackbar("Frames", "Detection",frame,this->framesCount,&Playback::onSlide,frame);
  cv::createTrackbar("Frames", "GT on RGB",frame,this->framesCount,&Playback::onSlide,frame);
  // this->livefeed = new std::queue <cv::Mat>;
  // this->inf_livefeed = new std::queue <cv::Mat>;
}

// Constructor if the framesCount is not known beforehand
Playback::Playback():framesCount(0),pause(false),speed(1),frameId(0),inferences(0),groundTruth(0){
  cv::namedWindow("Detection",cv::WINDOW_NORMAL);
  cv::namedWindow("GT on RGB",cv::WINDOW_NORMAL);
  // this->livefeed = new std::queue <cv::Mat>;
  // this->inf_livefeed = new std::queue <cv::Mat>;
}

// Adds trackbar to the Detection and undetected window
void Playback::AddTrackbar(long long int framesCount){
  int *frame = &this->frameId;
  this->framesCount = framesCount;
  cv::createTrackbar("Frames", "Detection",frame,this->framesCount,&Playback::onSlide,frame);
  cv::createTrackbar("Frames", "GT on RGB",frame,this->framesCount,&Playback::onSlide,frame);
}

// Check if paused
bool Playback::IsPaused(){
  return this->pause;
}

// Store the new frames into a vector , so that we can slide across them later
void Playback::GetInput(char keypressed , cv::Mat inference, cv::Mat groundTruth){
  this->inferences.push_back(inference);
  this->groundTruth.push_back(groundTruth);
  // Upate the keystroke with the new stroke
  this->keystroke = keypressed;
  // Call process
  Playback::process();
}

void Playback::GetInput(char keypressed ){
  this->keystroke = keypressed;
  Playback::process();
}

// After updating the keystroke , perform the actions accordingly
void Playback::process(){
  switch (this->keystroke) {
    // "Space" , "p" , "k" to pause the video
    case ' ' :
    case 'p' :
    case 'k' : this->pause = !this->pause;
               std::cout << "Keystroke : " << this->keystroke << std::endl;
               break;
    // '-' to reduce the video playback rate
    case '-' : this->speed += 2;
               std::cout << "Keystroke : " << this->keystroke << std::endl;
               break;
    // '-' to increase the video playback rate
    case '+' : if(this->speed>2)
                  this->speed-=2;
               std::cout << "Keystroke : " << this->keystroke << std::endl;
               break;
    default  : break;
  }
  Playback::show();
}

void Playback::show(){
  // If not paused , output the frame
  if(!this->pause){
    // Playback::WaitTillResume();
    usleep(int(this->rate()*10000));
    cv::imshow("Detection",this->inferences[this->frameId]);
    cv::imshow("GT on RGB",this->groundTruth[this->frameId]);
    // Update the frameID
    this->frameId++;
    cv::setTrackbarPos("Frames","Detection",this->frameId);
    cv::setTrackbarPos("Frames","GT on RGB",this->frameId);
  }
  // Else wait
  else
    Playback::WaitTillResume();
}

// Below function takes care once the video ends
void Playback::completeShow(){
  while(this->frameId!=this->inferences.size())
    Playback::GetInput(cv::waitKey(1));
}

// Current playback rate
double Playback::rate(){
  return this->speed;
}

// Wait till the video is resumed
void Playback::WaitTillResume(){
  while(this->pause)
    Playback::GetInput(cv::waitKey(0));
}

// Update the frame
void Playback::updateFrame(int frameId){
  this->frameId = frameId;
}

// Return current frameID
int Playback::currentFrame(){
  return this->frameId;
}

// Updates both the frameID and the image at that ID
void Playback::updateFrame(int FrameId ,cv::Mat *image){
  this->inferences.at(FrameId) = *image;
}
