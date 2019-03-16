/*
 * DepthFilter.cpp
 *
 *  Created on: 04/01/2014
 *      Author: frivas
 */

#include <opencv2/imgproc/imgproc.hpp>
#include "DepthFilter.h"
#include <glog/logging.h>

namespace jderobot {

DepthFilter::DepthFilter(){
	//default parameters
	this->type=3;
	this->buffer.resize(0);
	this->buffSize=7;
	this->erodeSize=0;
	this->threshold=10;
}

DepthFilter::DepthFilter(int type, int buffSize, int erodeSize, int threshold) {
	this->type=type;
	this->buffSize=buffSize;
	this->buffer.resize(0);
	this->erodeSize=erodeSize;
	this->threshold=threshold;
}

DepthFilter::~DepthFilter() {
	// TODO Auto-generated destructor stub
}
//filtrado bilateral, elimina ruido y mantiene los bordes
void
DepthFilter::filterDakkak(cv::Mat imageIn, cv::Mat& imageOut){

	int d=20;
	double sigmaColor=21;
    double sigmaSpace=3;

	std::vector<cv::Mat> layers;
	cv::split(imageIn, layers);
	cv::bilateralFilter(layers[0], imageOut, d,sigmaColor,sigmaSpace);
	cv::cvtColor(imageOut,imageOut,cv::COLOR_GRAY2RGB);
}

void
DepthFilter::filterMeanNonMotion3Channels(cv::Mat imageIn, cv::Mat& imageOut){
	cv::Mat localSource,meanNonZeroImage;

	cv::Mat zeros = cv::Mat(imageIn.rows,imageIn.cols, CV_8UC3, cv::Scalar(0,0,0));
	cv::Mat ones= cv::Mat(imageIn.rows,imageIn.cols, CV_8UC3, cv::Scalar(1,1,1));

	imageIn.copyTo(localSource);
	this->m.lock();
	/*for (int i=0; i< this->erodeSize;i++){
		cv::erode(localSource,localSource,cv::Mat());
	}
	for (int i=0; i< this->erodeSize;i++){
		cv::dilate(localSource,localSource,cv::Mat());
	}*/

	this->buffer.push_back(localSource);
	while ((int)this->buffer.size()>this->buffSize)
		this->buffer.pop_front();
	cv::Mat onesAcc; //acumulado de unos en al buffer
	cv::Mat resultAcc; //resultado acumulado

	zeros.copyTo(imageOut);
	zeros.copyTo(onesAcc);
	zeros.copyTo(resultAcc);

	resultAcc.convertTo(resultAcc,CV_32FC3);

	for ( std::list<cv::Mat>::iterator it=this->buffer.begin(); it!= this->buffer.end();it++){
		cv::Mat localOnes;

		cv::Mat localItFloat;
		it->convertTo(localItFloat,CV_32FC3);
		resultAcc=resultAcc+localItFloat;
		ones.copyTo(localOnes,*it);
		onesAcc=onesAcc+localOnes;
	}
	onesAcc.convertTo(onesAcc,CV_32FC3);
	resultAcc=resultAcc/onesAcc;
	resultAcc.copyTo(meanNonZeroImage);



	zeros.copyTo(onesAcc);
	zeros.copyTo(resultAcc);
	resultAcc.convertTo(resultAcc,CV_32FC3);
	cv::Mat accAndMask;
	cv::Mat localDiff, localMask;
	for ( std::list<cv::Mat>::iterator it=this->buffer.begin(); it!= this->buffer.end();it++){
		if (it==this->buffer.begin())
			it->copyTo(accAndMask);

		cv::Mat localOnes;
		cv::Mat localItFloat;
		it->convertTo(localItFloat,CV_32FC3);
		cv::absdiff(localItFloat,meanNonZeroImage,localDiff);
		localDiff.convertTo(localDiff,CV_8UC3);
		cv::threshold(localDiff,localMask,this->threshold, 255, cv::THRESH_BINARY_INV);
		resultAcc=resultAcc+localItFloat;
		localMask.convertTo(localMask,CV_8UC3);
		ones.copyTo(localOnes,localMask);
		onesAcc=onesAcc+localOnes;
		accAndMask=accAndMask & localMask;
	}
	onesAcc.convertTo(onesAcc,CV_32FC3);
	resultAcc=resultAcc/onesAcc;
	resultAcc.convertTo(resultAcc,CV_8UC3);

	resultAcc.copyTo(imageOut, accAndMask);
	zeros.copyTo(ones, accAndMask);

	imageIn.copyTo(imageOut,ones);

	//copio la parte sin movimiento del promedio de imágenes y la parte de movimiento de la última captura


		meanNonZeroImage.convertTo(this->globalMeanImage,CV_8UC3);
		localDiff.convertTo(this->globalDiffImage,CV_8UC3);

		accAndMask.copyTo(this->globalFirstMask);
		ones.copyTo(this->globalSecondMask);
	this->m.unlock();
}

//filtra únicamente las zonas de la imagen donde no hay movimiento
void DepthFilter::filterMeanNonMotion1Channels(cv::Mat imageIn, cv::Mat& imageOut){
	cv::Mat localSource,meanNonZeroImage;

	cv::Mat zeros3c = cv::Mat(imageIn.rows,imageIn.cols, CV_8UC3, cv::Scalar(0,0,0));
	cv::Mat ones3c= cv::Mat(imageIn.rows,imageIn.cols, CV_8UC3, cv::Scalar(1,1,1));
	cv::Mat zeros1c = cv::Mat(imageIn.rows,imageIn.cols, CV_8UC1, cv::Scalar(0));
	cv::Mat ones1c= cv::Mat(imageIn.rows,imageIn.cols, CV_8UC1, cv::Scalar(1));

	imageIn.copyTo(localSource);

	std::vector<cv::Mat> layers;
	cv::split(localSource, layers);

	/*for (int i=0; i< this->erodeSize;i++){
		cv::erode(localSource,localSource,cv::Mat());
	}
	for (int i=0; i< this->erodeSize;i++){
		cv::dilate(localSource,localSource,cv::Mat());
	}*/
	this->m.lock();
	this->buffer.push_back(localSource);
	this->bufferGray.push_back(layers[0]);
	while ((int)this->buffer.size()>this->buffSize)
		this->buffer.pop_front();
	while ((int)this->bufferGray.size()>this->buffSize)
		this->bufferGray.pop_front();
	cv::Mat onesAcc; //acumulado de unos en al buffer
	cv::Mat resultAcc; //resultado acumulado

	zeros3c.copyTo(imageOut);
	zeros3c.copyTo(onesAcc);
	zeros3c.copyTo(resultAcc);

	resultAcc.convertTo(resultAcc,CV_32FC1);

	//calculo la imagen promedio para luego hacer la comparación contra todas las imágenes individuales
	//y obtener las zonas comunes (sin movimiento y que podemos suavizar).
	for ( std::list<cv::Mat>::iterator it=this->buffer.begin(); it!= this->buffer.end();it++){
		cv::Mat localOnes;

		cv::Mat localItFloat;
		it->convertTo(localItFloat,CV_32FC3);
		resultAcc=resultAcc+localItFloat;
		ones3c.copyTo(localOnes,*it);
		onesAcc=onesAcc+localOnes;
	}
	onesAcc.convertTo(onesAcc,CV_32FC3);
	resultAcc=resultAcc/onesAcc;
	resultAcc.copyTo(meanNonZeroImage); //imagen promedio sin promediar zonas que son iguales a cero

	meanNonZeroImage.convertTo(meanNonZeroImage,CV_8UC3);

	//imagen promedio en escala de grises para el cálculo de las máscaras:
	std::vector<cv::Mat> layersMean;
	cv::split(meanNonZeroImage, layersMean);



	zeros1c.copyTo(onesAcc);
	zeros1c.copyTo(resultAcc);
	resultAcc.convertTo(resultAcc,CV_32FC1);
	cv::Mat accAndMask;

	ones1c.copyTo(accAndMask);
	cv::Mat localDiff, localMask;
	//comparamos todas las imagenes con el promedio y obtenemos zonas comunes
	for ( std::list<cv::Mat>::iterator it=this->bufferGray.begin(); it!= this->bufferGray.end();it++){

		cv::Mat localOnes;
		cv::absdiff(*it,layersMean[0],localDiff);
		localDiff.convertTo(localDiff,CV_8UC1);
		cv::threshold(localDiff,localMask,this->threshold, 255, cv::THRESH_BINARY_INV);
		localMask.convertTo(localMask,CV_8UC1);
		ones1c.copyTo(localOnes,localMask);
		accAndMask=accAndMask & localMask;
	}

	onesAcc.convertTo(onesAcc,CV_32FC3);
	zeros3c.copyTo(ones3c, accAndMask);
	for (int i=0; i< this->erodeSize;i++){
		cv::erode(accAndMask,accAndMask,cv::Mat());
		//cv::erode(ones3c,ones3c,cv::Mat());

	}
	for (int i=0; i< this->erodeSize;i++){
		cv::dilate(accAndMask,accAndMask,cv::Mat());
		//cv::dilate(ones3c,ones3c,cv::Mat());

	}

	meanNonZeroImage.copyTo(imageOut, accAndMask);
	imageIn.copyTo(imageOut,ones3c);

	//copio la parte sin movimiento del promedio de imágenes y la parte de movimiento de la última captura


		meanNonZeroImage.copyTo(this->globalMeanImage);
		cv::cvtColor(localDiff,globalDiffImage,cv::COLOR_GRAY2RGB);
		accAndMask=accAndMask*255;
		cv::cvtColor(accAndMask,this->globalFirstMask,cv::COLOR_GRAY2RGB);
		ones3c=ones3c*255;
		ones3c.copyTo(this->globalSecondMask);
	this->m.unlock();




}

//funcion que implementa el filtrado por media con valors no nulos
void
DepthFilter::filterMeanNonZero(cv::Mat imageIn, cv::Mat& imageOut){
	cv::Mat localSource,result;

	cv::Mat zeros = cv::Mat(imageIn.rows,imageIn.cols, CV_8UC3, cv::Scalar(0,0,0));
	cv::Mat ones= cv::Mat(imageIn.rows,imageIn.cols, CV_8UC3, cv::Scalar(1,1,1));

	imageIn.copyTo(localSource);

	for (int i=0; i< this->erodeSize;i++){
		cv::erode(localSource,localSource,cv::Mat());
	}
	for (int i=0; i< this->erodeSize;i++){
		cv::dilate(localSource,localSource,cv::Mat());
	}
	//localSource.convertTo(localSource,CV_32FC3);
	this->buffer.push_back(localSource);
	while ((int)this->buffer.size()>this->buffSize)
		this->buffer.pop_front();
	cv::Mat onesAcc; //acumulado de unos en al buffer
	cv::Mat resultAcc; //resultado acumulado

	zeros.copyTo(onesAcc);
	zeros.copyTo(resultAcc);

	resultAcc.convertTo(resultAcc,CV_32FC3);


	for ( std::list<cv::Mat>::iterator it=this->buffer.begin(); it!= this->buffer.end();it++){
		cv::Mat localOnes;

		cv::Mat localItFloat;
		it->convertTo(localItFloat,CV_32FC3);

		resultAcc=resultAcc+localItFloat;
		std::vector<cv::Mat> layers;
		//cv::split(*it, layers);
		ones.copyTo(localOnes,*it);
		onesAcc=onesAcc+localOnes;
	}
	onesAcc.convertTo(onesAcc,CV_32FC3);
	resultAcc=resultAcc/onesAcc;
	resultAcc.convertTo(resultAcc,CV_8UC3);
	resultAcc.copyTo(imageOut);

}


void DepthFilter::getMeanImage(cv::Mat& out){
	this->m.lock();
	this->globalMeanImage.copyTo(out);
	this->m.unlock();
}
void DepthFilter::getDiffImage(cv::Mat& out){
	this->m.lock();
	this->globalDiffImage.copyTo(out);
	this->m.unlock();
}
void DepthFilter::getFistMask(cv::Mat& out){
	this->m.lock();
	this->globalFirstMask.copyTo(out);
	this->m.unlock();
}
void DepthFilter::getSecondMask(cv::Mat& out){
	this->m.lock();
	this->globalSecondMask.copyTo(out);
	this->m.unlock();
}

void DepthFilter::clear(){
	this->m.lock();
	this->buffer.resize(0);
	this->bufferGray.resize(0);
	this->m.unlock();

};


//función iterativa que realimenta el filtrado
void
DepthFilter::update(cv::Mat imageIn, cv::Mat& imageOut){
	switch(type){
		case 0:
			filterMeanNonZero(imageIn,imageOut);
			break;
		case 1:
			filterDakkak(imageIn,imageOut);
			break;
		case 2:
			filterMeanNonMotion3Channels(imageIn, imageOut);
			break;
		case 3:
			filterMeanNonMotion1Channels(imageIn, imageOut);
			break;
		default:
			LOG(ERROR) << "Filter method: " << this->type << "not implemented" << std::endl;
			break;
	}
}

} /* namespace jderobot */
