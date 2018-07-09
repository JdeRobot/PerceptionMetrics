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

#ifndef JDEROBOTCOMM_CAMERACLIENT_H
#define JDEROBOTCOMM_CAMERACLIENT_H

#include <jderobottypes/image.h>
#include <colorspaces/colorspacesmm.h>
#include <comm/tools.hpp>
#include <comm/communicator.hpp>
#include <comm/interfaces/cameraClient.hpp>
#ifdef ICE
#include <Ice/Communicator.h>
#include <comm/ice/cameraIceClient.hpp>
#include <CameraUtils.h>
#endif
#ifdef JDERROS
#include <comm/ros/listenerCamera.hpp>
#endif





namespace Comm {

	/**
	 * @brief make a CameraClient using propierties
	 *
	 *
	 * @param communicator that contains properties
	 * @param prefix of client Propierties (example: "carViz.Camera")
	 *
	 *
	 * @return null if propierties are wrong
	 */
	CameraClient* getCameraClient(Comm::Communicator* jdrc, std::string prefix);


} //NS

#endif // JDEROBOTCOMM_CAMERACLIENT_H
