#include "DeployerNode.hpp"
#include <boost/bind.hpp>

DeployerNode::DeployerNode(int argc, char *argv[]){
  std::string path;
  ros::init(argc, argv, "deployer_node");
  ros::NodeHandle nh("~");
  nh.getParam("topic", this->topic);
  nh.getParam("configfile", path);

  ROS_INFO("Will subscribe to ROS TOPIC : %s", this->topic.c_str());

  YAML::Node config = YAML::LoadFile(path);
  const std::string &netConfigList   = (const std::string) config["netConfigList"].as<std::string>(),
                    &weights         = (const std::string) config["weights"].as<std::string>(),
                    &inferencerImp   = (const std::string) config["inferencerImp"].as<std::string>(),
                    &inferencerNames = (const std::string) config["inferencerNames"].as<std::string>(),
                    &outputFolder = "";
                    // &outputFolder    = (const std::string) config["outputFolder"].as<std::string>();
  // std::map<std::string, std::string> *inferencerParamsMap = config["outputFolder"].as<std::map<std::string, std::string>();
  std::map<std::string, std::string>* inferencerParamsMap = new std::map<std::string, std::string>();
  double* confidence_threshold = new double(0.2);
  this->inferencer = new GenericInferencer(netConfigList,weights,inferencerNames,inferencerImp, inferencerParamsMap);
  this->massInferencer = new MassInferencer(inferencer->getInferencer(),outputFolder, confidence_threshold, true);
  this->node = new ros::NodeHandle();
  this->sub = this->node->subscribe<sensor_msgs::Image>(this->topic, 10,boost::bind(&DeployerNode::ros_to_cv, _1, this));
  ros::NodeHandle *pub = new ros::NodeHandle();
  this->pub = pub->advertise <detection_suite::objects>("my_topic", 10);
  ros::spin();
}

void DeployerNode::ros_to_cv(const sensor_msgs::ImageConstPtr& img ,DeployerNode *node){
  cv_bridge::CvImagePtr cv_ptr;
  try{
    cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e){
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  cv::waitKey(3);
  node->massInferencer->process(false,cv_ptr->image);
  // RectRegionsPtr data = node->massInferencer->detections();
  // for (auto it = data->getRegions().begin(); it != data->getRegions().end(); it++){
    // node->detection.className = it->classID ;
  //   node->detection.confidence = it->confidence_score ;
  //   node->detection.x = it->region.x;
  //   node->detection.y = it->region.y;
  //   node->detection.height = it->region.height;
  //   node->detection.width = it->region.width;
    // node->detections.objects.push_back(node->detection);
  // }
  node->detections.objects.clear();
  Sample CurrFrame = node->massInferencer->getSample();
  CurrFrame.print();
  std::vector<RectRegion> regionsToPrint = CurrFrame.getRectRegions()->getRegions();
  for (auto it = regionsToPrint.begin(); it != regionsToPrint.end(); it++) {
        node->detection.className = it->classID ;//<< '\n';node->detection.confidence = it->confidence_score ;
        node->detection.confidence = it->confidence_score ;
        node->detection.x = it->region.x;
        node->detection.y = it->region.y;
        node->detection.height = it->region.height;
        node->detection.width = it->region.width;
        node->detections.objects.push_back(node->detection);
  }
  node->pub.publish(node->detections);
  // ros::spinOnce();
  // Sample frame = node->massInferencer->getSample();
  // frame.print();
}




DeployerNode::~DeployerNode(){
  delete inferencer;
  delete massInferencer;
  delete node;
}
