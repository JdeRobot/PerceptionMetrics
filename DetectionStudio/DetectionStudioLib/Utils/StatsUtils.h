//
// Created by frivas on 7/02/17.
//

#ifndef SAMPLERGENERATOR_STATSUTILS_H
#define SAMPLERGENERATOR_STATSUTILS_H

#include <opencv2/opencv.hpp>
#include <Common/Sample.h>
#include <Common/EvalMatrix.h>

class StatsUtils {
public:
    static double getIOU(const cv::Rect_<double> &gt, const cv::Rect_<double> &detection, bool isCrowd);
    static void computeIOUMatrix(Sample gt, Sample detection, Eval::EvalMatrix& evalmatrix, bool isIouTypeBbox);
};


#endif //SAMPLERGENERATOR_STATSUTILS_H
