/*
 *  Copyright (C) 1997-2016 JDE Developers Team
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see http://www.gnu.org/licenses/.
 *  Authors :
 *       Aitor Martinez Fernandez <aitor.martinez.fernandez@gmail.com>
 */

#ifndef JDEROBOTCOMM_LISTENERCAMERA_H_
#define JDEROBOTCOMM_LISTENERCAMERA_H_


#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <boost/thread/thread.hpp>
#include <jderobottypes/image.h>
#include <comm/interfaces/cameraClient.hpp>
#include <comm/ros/translators.hpp>
#include <time.h>

namespace Comm {
	class ListenerCamera: public Comm::CameraClient {

	public:
		ListenerCamera(int argc, char** argv, std::string nodeName, std::string topic);
		~ListenerCamera();

		void start();
		void stop();
		virtual JdeRobotTypes::Image getImage();
		virtual int getRefreshRate();


	private:
		pthread_mutex_t mutex;
		ros::Subscriber sub;
		std::string topic;
		std::string nodeName;

		int cont = 0; //used to count Frames per seconds
		time_t timer; // used to save time for FPS

		ros::AsyncSpinner* spinner;


		void imagecallback (const sensor_msgs::ImageConstPtr& image_msg);





	};//class

} //NS
#endif /* JDEROBOTCOMM_LISTENERCAMERA_H_ */
