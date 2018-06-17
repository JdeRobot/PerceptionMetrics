#include <comm/ros/translators.hpp>
namespace Comm {

	float PI = 3.1415;

	int MAXRANGEIMGD = 8; //max length received from imageD



	void
	depthToRGB(const cv::Mat& float_img, cv::Mat& rgb_img, std::string type ){
	  //Process images
		cv::Mat mono8_img;
		if (type.substr(type.length() - 3, 1) == "U"){
			mono8_img = float_img;
			rgb_img = cv::Mat(float_img.size(), CV_8UC3);
		}else{
			cv::Mat mono8_img = cv::Mat(float_img.size(), CV_8UC1);
		  	if(rgb_img.rows != float_img.rows || rgb_img.cols != float_img.cols){
		    	rgb_img = cv::Mat(float_img.size(), CV_8UC3);
		    }
		  	cv::convertScaleAbs(float_img, mono8_img, 255/MAXRANGEIMGD, 0.0);
		}

	  	cv::cvtColor(mono8_img, rgb_img, CV_GRAY2RGB);

	}


	JdeRobotTypes::Image
	translate_image_messages(const sensor_msgs::ImageConstPtr& image_msg){
		JdeRobotTypes::Image img;
		cv_bridge::CvImagePtr cv_ptr;

		img.timeStamp = image_msg->header.stamp.sec + (image_msg->header.stamp.nsec *1e-9);
		img.format = "RGB8"; // we convert img_msg to RGB8 format
		img.width = image_msg->width;
		img.height = image_msg->height;
		cv::Mat img_data;

		try {

			//std::cout << image_msg->encoding << std::endl;
			//if (image_msg->encoding.compare(sensor_msgs::image_encodings::TYPE_32FC1)==0 || image_msg->encoding.compare(sensor_msgs::image_encodings::TYPE_16UC1)==0){

			if (image_msg->encoding.substr(image_msg->encoding.length() - 2 ) == "C1"){
				cv_ptr = cv_bridge::toCvCopy(image_msg);
				depthToRGB(cv_ptr->image, img_data, image_msg->encoding);


			}else{
				cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::RGB8);
				img_data = 	cv_ptr->image;
			}
		} catch (cv_bridge::Exception& e) {

			ROS_ERROR("cv_bridge exception: %s", e.what());
		}

		img.data = img_data;

		return img;
	}

} /* NS */
