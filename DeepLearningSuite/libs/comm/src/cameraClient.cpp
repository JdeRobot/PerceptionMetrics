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
#include <comm/cameraClient.hpp>
#include <glog/logging.h>

namespace Comm {

CameraClient*
getCameraClient(Comm::Communicator* jdrc, std::string prefix){
	CameraClient* client = 0;

	int server;
	std::string server_name = jdrc->getConfig().asString(prefix+".Server");
	server = server2int(server_name);

	switch (server){
		case 0:
		{
			LOG(ERROR) << "Camera disabled" << std::endl;
			break;
		}
		case 1:
		{
			#ifdef ICE
			LOG(INFO) << "Receiving Image from ICE interfaces" << std::endl;
			CameraIceClient* cl;
			cl = new CameraIceClient(jdrc, prefix);
			cl->start();
		    client = (Comm::CameraClient*) cl;
            #else
                throw "ERROR: ICE is not available";
            #endif
		  	break;
		}
		case 2:
		{
			#ifdef JDERROS
				LOG(INFO) << "Receiving Image from ROS messages" << std::endl;
				std::string nodeName;
				nodeName =  jdrc->getConfig().asStringWithDefault(prefix+".Name", "LaserNode");
				std::string topic;
				topic = jdrc->getConfig().asStringWithDefault(prefix+".Topic", "");
				ListenerCamera* lc;
				lc = new ListenerCamera(0, nullptr, nodeName, topic);
				lc->start();
				client = (Comm::CameraClient*) lc;
            #else
				throw "ERROR: ROS is not available";
			#endif

		 	break;
		}
		default:
		{
			std::cerr << "Wrong " + prefix+".Server property" << std::endl;
			break;
		}

	}

	return client;


}

}//NS
