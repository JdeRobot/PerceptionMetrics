//
// Created by frivas on 7/02/17.
//

#include "StatsUtils.h"

double StatsUtils::getIOU(const cv::Rect_<double> &gt, const cv::Rect_<double> &detection, bool isCrowd) {
    //compute iou

    double xA = std::max(gt.x, detection.x);
    double yA = std::max(gt.y, detection.y);
	double xB = std::min(gt.x + gt.width, detection.x + detection.width);
	double yB = std::min(gt.y + gt.height, detection.y + detection.height);

    // computer area of intersection
    double interArea = ((xB - xA) > 0 ? (xB - xA) : 0 ) * ((yB - yA) > 0 ? (yB - yA) : 0);

	// compute the area of both the prediction and ground-truth
	// rectangles
	double boxAArea = (gt.width) * (gt.height);
	double boxBArea = (detection.width) * (detection.height);

	//compute the intersection over union by taking the intersection
	//area and dividing it by the sum of prediction + ground-truth
	//areas - the interesection area
    double iou;
    if (isCrowd) {
        iou = interArea / (boxBArea);
    } else {
        iou = interArea / (boxAArea + boxBArea - interArea);
    }


    //std::cout << gt.x << " " << gt.y << " " << gt.width << " " << gt.height << '\n';
    //std::cout << detection.x << " " << detection.y << " " << detection.width << " " << detection.height << '\n';
    //std::cout << imageSize << '\n';

    return iou;

    /*cv::Mat maskGT(imageSize, CV_8UC1, cv::Scalar(0));
    cv::Mat maskDetection(imageSize, CV_8UC1, cv::Scalar(0));

    cv::rectangle(maskGT, gt, cv::Scalar(255), -1);
    cv::rectangle(maskDetection, detection, cv::Scalar(255), -1);

    cv::Mat unionMask(imageSize, CV_8UC1, cv::Scalar(0));
    cv::rectangle(unionMask, gt, cv::Scalar(150), -1);
    cv::rectangle(unionMask, detection, cv::Scalar(255), -1);

    cv::Mat interSection = maskGT & maskDetection;

    int interSectionArea = cv::countNonZero(interSection);
    int unionArea = cv::countNonZero(unionMask);
    double iouValue = double(interSectionArea) / double(unionArea);
    return iouValue;*/
}

void StatsUtils::computeIOUMatrix(Sample gt, Sample detection, Eval::EvalMatrix& evalmatrix) {
    auto detectionRegions = detection.getRectRegions()->getRegions();
    auto gtRegions = gt.getRectRegions()->getRegions();

    std::string sampleID = gt.getSampleID();

    if (!evalmatrix.empty()) {
        throw std::runtime_error("EvalMatrix with sample ID isn't empty, Data might be duplicated while Evaluation");
    }

    // Sorting RectRegions by confidence_score for sam classID only
    // So, first of all it is necessary to segregate out RectRegions with
    // different classIds


    for (auto itDetection = detectionRegions.begin(); itDetection != detectionRegions.end(); ++itDetection) {

        std::string classID = itDetection->classID;
        std::vector<double> detectionIOUclassRow;

        for(auto itgt = gtRegions.begin(); itgt != gtRegions.end(); itgt++) {

            if (itgt->classID != classID) {
                continue;
            }

            double iouValue;
            //std::cout << itDetection->classID << " " << itDetection->confidence_score <<'\n';
            iouValue = StatsUtils::getIOU(itgt->region, itDetection->region, itgt->isCrowd);
            //std::cout << "Bbox Gt: " << itgt->region.x << " " << itgt->region.y << " " << itgt->region.width << " " << itgt->region.height << '\n';
            //std::cout << "Bbox Dt: " << itDetection->region.x << " " << itDetection->region.y << " " << itDetection->region.width << " " << itDetection->region.height << '\n';
            //std::cout << iouValue << '\n';
            detectionIOUclassRow.push_back(iouValue);

        }

        evalmatrix[classID].push_back(detectionIOUclassRow);

    }

    //std::cout << "Matrix Computed" << '\n';
}
