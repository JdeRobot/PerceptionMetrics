/*
 * DepthSampler.h
 *
 *  Created on: 08/01/2014
 *      Author: eldercare
 */

#ifndef DEPTHSAMPLER_H_
#define DEPTHSAMPLER_H_

#include <cv.h>
#include <highgui.h>
#include <list>
#include <IceUtil/Mutex.h>
#include <numeric>
namespace jderobot {

class DepthSampler {
public:
	DepthSampler(int nBins, int maxDistance, int minInd, float step);
	DepthSampler();
	virtual ~DepthSampler();
	void calculateLayers(cv::Mat source, std::vector<cv::Mat>& layers);
	void evalSample(cv::Mat source, std::vector<cv::Mat> layers, int samplingRate, cv::Mat &outSNormal, cv::Mat &outSLayers);
	void sample(cv::Mat source, std::vector<cv::Mat> layers, std::vector<cv::Point2i>& out);

	void setnBins(int value){this->nBins=value;};
	int getnBins(){return this->nBins;};
	void setMaxDistance(int value){this->maxDistance=value;};
	int getMaxDistance(){return this->maxDistance;};
	void setMinInd(int value){this->minInd=value;};
	int getMinInd(){return this->minInd;};
	void setStep(double value){this->step=value; std::cout << "SETTING STEP TO: " << value << std::endl;};
	double getStep(){return this->step;};



private:
	int nBins, maxDistance, minInd;
	double step;

};

} /* namespace jderobot */

#endif /* DEPTHSAMPLER_H_ */
