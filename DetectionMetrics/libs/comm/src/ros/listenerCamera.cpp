#include <comm/ros/listenerCamera.hpp>
#include <glog/logging.h>
namespace Comm {

	ListenerCamera::ListenerCamera(int argc, char** argv, std::string nodeName, std::string topic){
		pthread_mutex_init(&mutex, NULL);
		if ("" == topic){
			this->on = false;
			std::cerr <<"Invalid camera topic" <<std::endl;
		}else{
			this->on = true;
			this->topic = topic;
			this->nodeName = nodeName;

			const std::string name = std::string(this->nodeName);

			time(&timer);
			int a = 0;
			ros::init(a, nullptr, name);
			ros::NodeHandle nh;
			this->sub = nh.subscribe(this->topic, 1001, &ListenerCamera::imagecallback, this);
			LOG(INFO) << "listen from "+ this->topic << std::endl;

			this->spinner = new ros::AsyncSpinner(1);
		}
	}



	ListenerCamera::~ListenerCamera(){
		this->stop();
	}

	void
	ListenerCamera::start(){
		this->spinner->start();
	}

	void
	ListenerCamera::stop(){
		this->spinner->stop();
		ros::shutdown();
	}

	void
	ListenerCamera::imagecallback(const sensor_msgs::ImageConstPtr& image_msg){
		this->cont++;
		time_t now;
		time(&now);
		pthread_mutex_lock(&mutex);
		this->image = Comm::translate_image_messages(image_msg);
		if (difftime(this->timer, now)>=1){
			this->refreshRate = this->cont;
			this->cont = 0;
			this->timer = now;
		}
		pthread_mutex_unlock(&mutex);

	}

	JdeRobotTypes::Image  ListenerCamera::getImage(){
		JdeRobotTypes::Image img;
		pthread_mutex_lock(&mutex);
		img = this->image;
		pthread_mutex_unlock(&mutex);
		return img;
	}

	int ListenerCamera::getRefreshRate(){

		int rr;
		pthread_mutex_lock(&mutex);
		rr = this->refreshRate;
		pthread_mutex_unlock(&mutex);

		return rr;
	}



}//NS
