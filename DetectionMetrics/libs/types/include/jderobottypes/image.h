/*
 *  Copyright (C) 1997-2016 JDE JdeRobot Developers Team
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

#ifndef JDEROBOTTYPES_IMAGE_H
#define JDEROBOTTYPES_IMAGE_H

#include <opencv2/core/core.hpp>

namespace JdeRobotTypes {

	class Image {
	public:

	    int height = 0; /**< %Image height [pixels] */
	    int width = 0; /**< %Image width [pixels] */
	    double timeStamp = 0; /**< %Time stamp [s] */
	    std::string format = ""; /**< %Image format string (RGB8, BGR,...) */
	    cv::Mat data = cv::Mat::zeros(3,3, CV_8UC3); /**< The image data itself */
	};


} //NS JdeRobotTypes

#endif // JDEROBOTTYPES_IMAGE_H
