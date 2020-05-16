#ifndef DEPLOYER_NODE
#define DEPLOYER_NODE
#include "std_msgs/String.h"
#include "DetectionStudioROS/objects.h"
#include "DetectionStudioROS/object.h"

#include <typeinfo>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <FrameworkEvaluator/GenericInferencer.h>
#include <FrameworkEvaluator/MassInferencer.h>
#include <FrameworkEvaluator/Labelling.h>
#include <Regions/RectRegions.h>
#include <yaml-cpp/yaml.h>

class DeployerNode{
public:
  DeployerNode(int argc, char *argv[]);
  ~DeployerNode();
  static void ros_to_cv(const sensor_msgs::ImageConstPtr &ros_img , DeployerNode *node);

private:
  cv::Mat cv_frame;
  ros::NodeHandle *node;
  ros::Subscriber sub;
  ros::Publisher pub;
  GenericInferencer *inferencer;
  MassInferencer *massInferencer;
  std::string topic;
  DetectionStudioROS::object detection;
  DetectionStudioROS::objects detections;
};

#endif
