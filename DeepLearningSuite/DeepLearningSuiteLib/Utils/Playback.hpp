#ifndef PLAYBACK_RATES
#define PLAYBACK_RATES
// Code for controlling playback functionalities
#include<iostream>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <unistd.h>

class Playback {
public:
  Playback(long long int framesCount);
  Playback();
  void GetInput(char keypressed , cv::Mat inference, cv::Mat groundTruth);
  void GetInput(char keypressed);
  void AddTrackbar(long long int framesCount);
  bool IsPaused();
  void WaitTillResume();
  void process();
  double rate();
  void show();
  void completeShow();
  void updateFrame(int frameId);
  void updateFrame(int FrameId, cv::Mat *image);
  int currentFrame();
  static void onSlide(int currentPos,void *frame);
private:
  bool pause;
  double speed;
  char keystroke;
  std::vector<cv::Mat> inferences;
  std::vector<cv::Mat> groundTruth;
  int frameId;
  long long int framesCount;
};
#endif
