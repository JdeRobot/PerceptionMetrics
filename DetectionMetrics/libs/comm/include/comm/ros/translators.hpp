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

#ifndef JDEROBOTCOMM_TRANSLATORSROS_H_
#define JDEROBOTCOMM_TRANSLATORSROS_H_

#include <ros/ros.h>
#include <cv.h>

#include <vector>

#include <jderobottypes/image.h>
#include <jderobottypes/rgbd.h>
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.h"


namespace Comm {

	/**
	 * @brief translate ROS Image messages to JdeRobot Image
	 *
	 *
	 * @param ROS Image Message
	 *
	 *
	 * @return Image translated from ROS Message
	 */
	JdeRobotTypes::Image translate_image_messages(const sensor_msgs::ImageConstPtr& image_msg);


	/**
	 * @brief translate ROS images messages to JdeRobot Rgbd
	 *
	 *
	 * @param ROS Image Message
	 * @param ROS Image Message
	 *
	 *
	 * @return Rgbd translated from ROS Messages
	 */
	JdeRobotTypes::Rgbd translate_rgbd(const sensor_msgs::ImageConstPtr& rgb,const sensor_msgs::ImageConstPtr& d);

	/**
	 * @brief Translates from 32FC1 Image format to RGB. Inf values are represented by NaN, when converting to RGB, NaN passed to 0
	 *
	 *
	 * @param ROS Image Message
	 *
	 *
	 * @return Image translated from ROS Message
	 */
	void depthToRGB(const cv::Mat& float_img, cv::Mat& rgb_img);

} /* NS */
#endif //JDEROBOTCOMM_TRANSLATORSROS_H_
