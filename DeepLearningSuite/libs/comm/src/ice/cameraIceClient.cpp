/*
 *  Copyright (C) 1997-2016 JDE Developers TeamkinectViewer.camRGB
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

#include <comm/ice/cameraIceClient.hpp>
#include <glog/logging.h>
#include <exception>


namespace Comm {


CameraIceClient::CameraIceClient(Comm::Communicator* jdrc, std::string prefix) {

	this->prefix=prefix;
	Ice::ObjectPrx baseCamera;
	this->refreshRate=0;
	this->mImageFormat.empty();



	float fps=jdrc->getConfig().asFloatWithDefault(prefix+".Fps", 30);
	this->cycle=(1/fps)*1000000;
	try{
		std::string proxy = jdrc->getConfig().asString(prefix+".Proxy");
		baseCamera = jdrc->getIceComm()->stringToProxy(proxy);
		if (0==baseCamera){
			this->on = false;
			throw prefix + "Could not create proxy with Camera";
		}
		else {
			this->prx= jderobot::CameraPrx::checkedCast(baseCamera);
			this->on = true;
			if (0==this->prx){
				this->on = false;
				throw "Invalid " + prefix + ".Proxy";
			}
		}
	}catch (const Ice::Exception& ex) {
		LOG(ERROR) << ex << std::endl;
	}
	catch (const char* msg) {
		LOG(ERROR) << msg << std::endl;
		LOG(ERROR) <<prefix + " Not camera provided";
	}

	//check if default format is defined
	std::string definedFormat=jdrc->getConfig().asStringWithDefault(prefix+".Format", "RGB8");

	this->mImageFormat = CameraUtils::negotiateDefaultFormat(this->prx,definedFormat);

	jderobot::ImageDataPtr data = this->prx->getImageData(this->mImageFormat);

	this->pauseStatus=false;
}


CameraIceClient::~CameraIceClient() {
	this->on=false;
}


jderobot::ImageFormat CameraIceClient::getImageFormat()
{
	return (this->prx->getImageFormat());
}

void CameraIceClient::setImageFormat (std::string format)
{
	mImageFormat = format;

	LOG(INFO) <<"Changed format " + this->mImageFormat + " for camera " + this->prx->getCameraDescription()->name;
};


void CameraIceClient::reset(){
	this->prx->reset();
}

void CameraIceClient::pause(){
	this->pauseStatus=true;
}

void CameraIceClient::resume(){
	this->controlMutex.lock();
	this->pauseStatus=false;
	this->semWait.broadcast();
	this->controlMutex.unlock();
}


void
CameraIceClient::run(){
	jderobot::ImageDataPtr dataPtr;
	colorspaces::Image::FormatPtr fmt;
	IceUtil::Time last;

	int iterIndex = 0;
	int totalRefreshRate = 0;
	int refrRate = 0;

	JdeRobotTypes::Image img;

	last=IceUtil::Time::now();
	while (this->on){

		iterIndex ++;
		if (pauseStatus){
			IceUtil::Mutex::Lock sync(this->controlMutex);
			this->semWait.wait(sync);
		}

		try{


			dataPtr = this->prx->getImageData(this->mImageFormat);





			// Putting image data
			img.data = CameraUtils::getImageFromCameraProxy(dataPtr);

			img.format = dataPtr->description->format;
			img.width = dataPtr->description->width;
			img.height = dataPtr->description->height;
			img.timeStamp = dataPtr->timeStamp.seconds + dataPtr->timeStamp.useconds * 1e-6;




		}
		catch(std::exception& e){
			LOG(WARNING) << prefix +"error during request (connection error): " << e.what() << std::endl;
			usleep(50000);

		}

		int process = (IceUtil::Time::now().toMicroSeconds() - last.toMicroSeconds());



		if (process > (int)cycle ){
			DLOG(WARNING) << "--------" + prefix + " adquisition timeout-";
		}
		else{
			int delay = (int)cycle - process;
			if (delay <1 || delay > (int)cycle)
				delay = 1;

			usleep(delay);
		}


		int rate =(int)(1000000/(IceUtil::Time::now().toMicroSeconds() - last.toMicroSeconds()));
		totalRefreshRate =  totalRefreshRate + rate;
		refrRate = totalRefreshRate / iterIndex;
		last=IceUtil::Time::now();

		if (iterIndex == INT_MAX)
		{
			iterIndex = 0;
			DLOG(INFO) << "*** Counter reset";
		}

		this->controlMutex.lock();
		this->image = img;
		this->refreshRate = refrRate;
		this->controlMutex.unlock();

	}

	this->image.data.release();
}

void CameraIceClient::stop_thread()
{
	this->on=false;
}

JdeRobotTypes::Image CameraIceClient::getImage(){
	JdeRobotTypes::Image img;

	this->controlMutex.lock();
	img = this->image;
	this->controlMutex.unlock();

	return img;
}

int CameraIceClient::getRefreshRate(){
	int rr;
	this->controlMutex.lock();
	rr = this->refreshRate;
	this->controlMutex.unlock();

	return rr;
};

} /* namespace jderobot */
