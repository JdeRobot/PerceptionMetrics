/*
 * DepthSampler.cpp
 *
 *  Created on: 08/01/2014
 *      Author: frivas
 */

#include <opencv2/imgproc/imgproc.hpp>
#include "DepthSampler.h"
#include <glog/logging.h>

namespace jderobot {
DepthSampler::DepthSampler(int nBins, int maxDistance, int minInd, float step){
	this->nBins=nBins;
	this->maxDistance=maxDistance;
	this->minInd=minInd;
	this->step=step;
}

DepthSampler::DepthSampler() {
	this->nBins=1;
	this->maxDistance=10000;
	this->minInd=3;
	this->step=1;
}

DepthSampler::~DepthSampler() {
	// TODO Auto-generated destructor stub
}

//devuelve en out, los nBins layers con la profundidad discretizada hasta maxDistance
void DepthSampler::calculateLayers(cv::Mat source, std::vector<cv::Mat>& layers){
	std::vector<cv::Mat> imgLayers;
	cv::Mat localSource;

	layers.resize(0);

	source.convertTo(localSource,CV_32FC3);
	cv::split(localSource, imgLayers);
	cv::Mat dM(localSource.rows,localSource.cols, CV_32FC1);
	dM=(imgLayers[1]*256) + imgLayers[2];

	cv::Mat unosc1(localSource.rows, localSource.cols, CV_8UC1, cv::Scalar(255));
	cv::Mat zerosc1(localSource.rows, localSource.cols, CV_8UC1, cv::Scalar(0));

	for (int i=0; i<nBins; i++){
		cv::Mat localMask;
		cv::threshold(dM,localMask,(maxDistance/nBins)*(i+1), 255, cv::THRESH_BINARY_INV);
		localMask.convertTo(localMask,CV_8UC1);
		layers.push_back(localMask&unosc1);
		zerosc1.copyTo(unosc1,localMask);
	}
}

//compara el muestreo normal con el muestreo por capas, empezando desde minInd, e incrementando step en cada capa,
//desde la mas lejana hasta la mas cercana.
void DepthSampler::evalSample(cv::Mat source, std::vector<cv::Mat> layers, int samplingRate, cv::Mat &outSNormal, cv::Mat &outSLayers){
	cv::Mat imgNormalSample=cv::Mat(source.rows,source.cols, CV_8UC1, cv::Scalar(0));
	cv::Mat imgNlayerSample=cv::Mat(source.rows,source.cols, CV_8UC1, cv::Scalar(0));;

	std::vector<int> normalSample;
	int outlierNormal=0;
	std::vector<int> layerSample;

	int nLayers=layers.size();
	normalSample.resize(nLayers);
	layerSample.resize(nLayers);

	cv::Mat localSource;
	std::vector<cv::Mat> tempLayers;
	source.convertTo(localSource,CV_32FC3);
	cv::split(localSource, tempLayers);
	cv::Mat dM(localSource.rows,localSource.cols, CV_32FC1);
	dM=(tempLayers[1]*256) + tempLayers[2];


	for (int i=0; i< nLayers;i++){
		normalSample[i]=0;
		layerSample[i]=0;
	}


	int start_s=clock();
	for (int xIm=0; xIm< source.cols; xIm+=samplingRate) {
		for (int yIm=0; yIm<source.rows ; yIm+=samplingRate) {
			float d=dM.at<float>(yIm,xIm);
			if (d != 0){
				imgNormalSample.at<char>(yIm,xIm)=(char)255;
				double pos= d/maxDistance;
				if (pos>1){
					outlierNormal++;
				}
				else{
					normalSample[floor(pos*nLayers)]=normalSample[floor(pos*nLayers)]++;
				}
			}
		}
	}

	int stop_s=clock();
	LOG(INFO) << "Time for normal sampling: " << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000 << std::endl;
	LOG(INFO) << "Layer sampling Size : " << std::accumulate(normalSample.begin(), normalSample.end(), 0) << ",  result:" << std::endl;
	for (std::vector<int>::iterator it= normalSample.begin(); it!= normalSample.end(); it++){
		LOG(INFO) << *it << std::endl;
	}
	start_s=clock();

	float localStep=minInd+(step*nLayers);
	for ( std::vector<cv::Mat>::iterator it= layers.begin(); it != layers.end(); it++){
		LOG(INFO) << "step: " << (int)localStep << std::endl;
		for (int xIm=0; xIm< source.cols; xIm+=(int)localStep) {
			for (int yIm=0; yIm<source.rows ; yIm+=(int)localStep) {
				if ((int)it->at<char>(yIm,xIm) != 0){
					float d=dM.at<float>(yIm,xIm);

					if (d != 0){
						imgNlayerSample.at<char>(yIm,xIm)=(char)255;
						double pos= d/maxDistance;
						if (pos>1){
							outlierNormal++;
						}
						else{
							layerSample[floor(pos*nLayers)]=layerSample[floor(pos*nLayers)]++;
						}
					}
				}
			}
		}

		localStep=localStep-step;
	}

	stop_s=clock();
	LOG(INFO) << "Time for layers sampling: " << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000 << std::endl;
	LOG(INFO) << "Layer sampling Size : " << std::accumulate(layerSample.begin(), layerSample.end(), 0) << ",  result:" << std::endl;
	for (std::vector<int>::iterator it= layerSample.begin(); it!= layerSample.end(); it++){
		LOG(INFO) << *it << std::endl;
	}

	/*cv::imshow("normal", imgNormalSample);
	cv::imshow("layers", imgNlayerSample);
	cv::waitKey(0);*/
	imgNormalSample.copyTo(outSNormal);
	imgNlayerSample.copyTo(outSLayers);
}

void DepthSampler::sample(cv::Mat source, std::vector<cv::Mat> layers, std::vector<cv::Point2i>& out){

	cv::Mat localSource;
	std::vector<cv::Mat> tempLayers;
	source.convertTo(localSource,CV_32FC3);
	cv::split(localSource, tempLayers);
	cv::Mat dM(localSource.rows,localSource.cols, CV_32FC1);
	dM=(tempLayers[1]*256) + tempLayers[2];
	out.resize(0);

	int nLayers=layers.size();
	float localStep=minInd+(step*(double)nLayers);
	for ( std::vector<cv::Mat>::iterator it= layers.begin(); it != layers.end(); it++){
		for (int xIm=0; xIm< source.cols; xIm+=(int)localStep) {
			for (int yIm=0; yIm<source.rows ; yIm+=(int)localStep) {
				if ((int)it->at<char>(yIm,xIm) != 0){
					float d=dM.at<float>(yIm,xIm);
					if ((d != 0)&&(d<maxDistance)){
						out.push_back(cv::Point2i(xIm,yIm));
					}
				}
			}
		}
		localStep=localStep-step;
	}
}


} /* namespace jderobot */
