/*
 *  Copyright (C) 1997-2013 JDE Developers TeamkinectViewer.camRGB
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
 *
 *  Author : Jose María Cañas <jmplaza@gsyc.es>
			Francisco Miguel Rivas Montero <franciscomiguel.rivas@urjc.es>

 */

#ifndef JDEROBOTCOMM_CAMERAICECLIENT_H_
#define JDEROBOTCOMM_CAMERAICECLIENT_H_

#include <IceUtil/IceUtil.h>
#include <Ice/Ice.h>
#include <cv.h>
#include <image.h>
#include <comm/communicator.hpp>
#include <CameraUtils.h>
#include <colorspaces/colorspacesmm.h>
#include <comm/interfaces/cameraClient.hpp>
#include <zlib.h>
#include <jderobot/camera.h>
#include <jderobot/image.h>

namespace Comm {

class CameraIceClient: public IceUtil::Thread, public Comm::CameraClient {
public:
	CameraIceClient(Comm::Communicator* jdrc, std::string prefix);
	virtual ~CameraIceClient();
	virtual void run();

	//callbacks
	virtual JdeRobotTypes::Image getImage();
	virtual int getRefreshRate();


	void pause();
	void resume();
	void reset();
	void stop_thread();
	bool getPause(){return pauseStatus;};

	jderobot::ImageFormat getImageFormat();
	void setImageFormat (std::string format);


private:
	jderobot::CameraPrx prx;
	long long int cycle;

	IceUtil::Mutex controlMutex;
	std::string prefix;

	bool pauseStatus;

	IceUtil::Cond semWait;
	std::string mImageFormat;

};

} /* namespace Comm */
#endif /* JDEROBOTCOMM_CAMERAICECLIENT_H_ */
