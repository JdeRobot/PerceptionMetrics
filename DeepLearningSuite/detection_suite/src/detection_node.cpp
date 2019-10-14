#include "std_msgs/String.h"

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <FrameworkEvaluator/GenericInferencer.h>
#include <FrameworkEvaluator/MassInferencer.h>
#include <FrameworkEvaluator/labeling.h>
#include <yaml-cpp/yaml.h>


GenericInferencer *inferencer;
MassInferencer *massInferencer;


void chatterCallback(const sensor_msgs::ImageConstPtr& img){

  cv_bridge::CvImagePtr cv_ptr;
  try{
    cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e){
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // cv::imshow("Mat", cv_ptr->image);
  cv::waitKey(3);

  massInferencer->process(false,cv_ptr->image);

}

void chatterCallback(const cv::Mat& img){
  massInferencer->process(false,img);
}


int main(int argc, char *argv[]){
  std::string path,topic;
  ros::init(argc, argv, "deployer_node");
  ros::NodeHandle nh("~");
  nh.getParam("topic", topic);
  nh.getParam("configfile", path);

  ROS_INFO("Will subscribe to ROS TOPIC : %s", topic.c_str());
  ROS_INFO("Will subscribe to ROS TOPIC : %s", path.c_str());

  YAML::Node config = YAML::LoadFile(path);
  std::cout << config["netConfigList"].as<std::string>() << std::endl;
  const std::string &netConfigList   = (const std::string) config["netConfigList"].as<std::string>(),
                    &inferencerNames = (const std::string) config["inferencerNames"].as<std::string>(),
                    &weights         = (const std::string) config["weights"].as<std::string>(),
                    &inferencerImp   = (const std::string) config["inferencerImp"].as<std::string>(),
                    &outputFolder = "";

  // std::string temp = config["netConfigList"].as<std::string>();
  // if(config["outputFolder"])
  //   temp = config["outputFolder"].as<std::string>() ;
  // else
  //   temp = "";
    // const std::string &outputFolder = (const std::string )temp;
  // std::map<std::string, std::string> *inferencerParamsMap = config["outputFolder"].as<std::map<std::string, std::string>();
  std::map<std::string, std::string>* inferencerParamsMap = new std::map<std::string, std::string>();
  double* confidence_threshold = new double(0.2);
  inferencer = new GenericInferencer(netConfigList,weights,inferencerNames,inferencerImp, inferencerParamsMap);
  massInferencer = new MassInferencer(inferencer->getInferencer(),outputFolder, confidence_threshold, true);
  ros::NodeHandle n;
  ros::Subscriber sub = n.subscribe(topic, 10, chatterCallback);

  ros::spin();

  return 0;
}
