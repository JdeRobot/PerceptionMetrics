/*
 * DepthFilter.h
 *
 *  Created on: 04/01/2014
 *      Author: frivas
 */

#ifndef DEPTHFILTER_H_
#define DEPTHFILTER_H_

#include <opencv2/opencv.hpp>
#include <list>
#include <mutex>
#include <iostream>

namespace jderobot {

class DepthFilter {
public:
	DepthFilter(int type, int buffSize, int erodeSize, int threshold);
	DepthFilter();
	virtual ~DepthFilter();
	void update(cv::Mat imageIn, cv::Mat& imageOut);
	int getBufferSize(){return this->buffSize;};
	void setBufferSize(int size){ this->buffSize=size;};
	int getErodeSize(){return this->erodeSize;};
	void setErodeSize(int size){ this->erodeSize=size;};
	int getThreshold(){return this->threshold;};
	void setThreshold(int value){this->threshold=value;};
	void setFilterType(int value){this->type=value;};
	int getFilterType(){return this->type;};
	void clear();
	void getMeanImage(cv::Mat& out);
	void getDiffImage(cv::Mat& out);
	void getFistMask(cv::Mat& out);
	void getSecondMask(cv::Mat& out);

private:
	int type; //tipo de filtro a aplicar
	int buffSize; //tamaño del buffer de imágenes
	std::list<cv::Mat> buffer; //buffer de imágenes
	std::list<cv::Mat> bufferGray; //buffer de imágenes en escala de grises
	int erodeSize; //numero de erosiones a realizar en el filtrado
	int threshold; //umbral para el filtrado
	std::mutex m; //control de datos compartidos

	//callbacks
	void filterMeanNonZero(cv::Mat imageIn, cv::Mat& imageOut);
	void filterDakkak(cv::Mat imageIn, cv::Mat& imageOut);
	void filterMeanNonMotion3Channels(cv::Mat imageIn, cv::Mat& imageOut);
	void filterMeanNonMotion1Channels(cv::Mat imageIn, cv::Mat& imageOut);

	//tempImages
	cv::Mat globalMeanImage;
	cv::Mat globalDiffImage;
	cv::Mat globalFirstMask;
	cv::Mat globalSecondMask;

};

} /* namespace jderobot */

#endif /* DEPTHFILTER_H_ */
