//
// Created by frivas on 1/02/17.
//

#include <glog/logging.h>
#include <Utils/StatsUtils.h>
#include "DetectionsEvaluator.h"


DetectionsEvaluator::DetectionsEvaluator(DatasetReaderPtr gt, DatasetReaderPtr detections,bool debug):
        gt(gt),detections(detections),debug(debug) {
    thIOU = 0.5;

    for (int i = 0; i < 101; i++) {
        this->recallThrs.push_back(0.01*i);
    } // Initializing Recall Thersholds with 101 values starting
      // from 0 with a difference of 0.01.

}

std::map<std::string, double> DetectionsEvaluator::getClassWiseAP() {
    return this->classWiseMeanAP;
}

std::map<std::string, double> DetectionsEvaluator::getClassWiseAR() {
    return this->classWiseMeanAR;
}

double DetectionsEvaluator::getOverallmAP() {
    return this->ApDiffIou.sum()/10;
}

double DetectionsEvaluator::getOverallmAR() {
    return this->ArDiffIou.sum()/10;
}

double DetectionsEvaluator::getEvaluationTime() {
    return this->timeEvaluation;
}

double DetectionsEvaluator::getAccumulationTime() {
    return this->timeAccumulation;
}

void DetectionsEvaluator::accumulateResults() {

    int start_s=clock();

    unsigned int index = 0;
    for (auto itr = this->sampleStats.begin(); itr != this->sampleStats.end(); itr++) {

        std::map<std::string, ClassStatistics> mystats = itr->second.getStats();

        int totalCount = 0;
        double totalPrecision = 0;
        double totalRecall = 0;

        int classCount = 0;

        for (auto iter = mystats.begin(); iter != mystats.end(); iter++) {

            if (iter->second.numGroundTruthsReg == 0)
                continue;

            std::vector<double> pr = iter->second.getPrecisionArray();
            std::vector<double> rc = iter->second.getRecallArray();

            double recall = 0;


            recall = iter->second.getRecall();

            this->classWiseMeanAR[iter->first] += recall / 10;            // 10 IOU Thresholds,
                                                                    // mean will be calculated directly
            totalRecall += recall;


            double precision = iter->second.getAveragePrecision(this->recallThrs);
            this->classWiseMeanAP[iter->first] += precision / 10;     // 10 IOU Thresholds,
                                                                // mean will be calculated directly
            totalPrecision += precision;

            totalCount++;


        }
        this->ApDiffIou[index] = totalPrecision / totalCount;
        this->ArDiffIou[index] = totalRecall / totalCount;

        ++index;
    }

    int stop_s=clock();
    this->timeAccumulation = (stop_s-start_s)/double(CLOCKS_PER_SEC);
    LOG(INFO) << "Time Taken in Accumulation: " << this->timeAccumulation << " seconds" << std::endl;
    LOG(INFO) << std::fixed;
    LOG(INFO) << std::setprecision(8);


    for (int i = 0; i < this->ApDiffIou.size(); i++) {
        LOG(INFO) << "AP for IOU " << this->iouThrs[i] <<  ": \t" << this->ApDiffIou[i] << '\n';
    }

    for (int i = 0; i < this->ArDiffIou.size(); i++) {
        LOG(INFO) << "AR for IOU " << this->iouThrs[i] <<  ": \t" << this->ArDiffIou[i] << '\n';
    }

    LOG(INFO) << "AP for IOU 0.5:0.95 \t" << this->ApDiffIou.sum()/10 << '\n';

    LOG(INFO) << "AR for IOU 0.5:0.95 \t" << this->ArDiffIou.sum()/10 << '\n';

    cv::destroyAllWindows();

    LOG(INFO) << "Evaluated Successfully" << '\n';

}

void DetectionsEvaluator::evaluate(bool isIouTypeBbox) {
    int counter=-1;
    int gtSamples = this->gt->getNumberOfElements();
    int detectionSamples = this->detections->getNumberOfElements();


    int start_s=clock();

    if (gtSamples != detectionSamples){
        LOG(WARNING) << "Both dataset has not the same number of elements";
    }

    ClassTypeMapper classMapper(this->gt->getClassNamesFile());
    this->classMapping = classMapper.mapFile(this->detections->getClassNamesFile());


    Sample gtSample;
    Sample detectionSample;


    while (this->gt->getNextSample(gtSample)) {
        counter++;


        this->detections->getNextSample(detectionSample);


        LOG(INFO) << "Evaluating: " << detectionSample.getSampleID() << "(" << counter << "/" << gtSamples << ")" << std::endl;


        if (gtSample.getSampleID().compare(detectionSample.getSampleID()) != 0){
            LOG(WARNING) << "No detection sample available, Creating Dummy Sample\n";
            Sample dummy;
            dummy.setSampleID(gtSample.getSampleID());
            dummy.setColorImage(gtSample.getColorImagePath());
            evaluateSample(gtSample, dummy, isIouTypeBbox);
            this->detections->decrementReaderCounter();
            const std::string error="Both dataset has not the same structure ids mismatch from:" + gtSample.getSampleID() + " to " + detectionSample.getSampleID();
            LOG(WARNING) << error;

        } else {
            evaluateSample(gtSample,detectionSample, isIouTypeBbox);
        }

        /*if (this->debug){
            cv::imshow("GT", gtSample.getSampledColorImage());
            cv::imshow("Detection", detectionSample.getSampledColorImage());
            cv::waitKey(10);
        }*/

    }
    int stop_s=clock();
    this->timeEvaluation = (stop_s-start_s)/double(CLOCKS_PER_SEC);
    LOG(INFO) << "Time Taken in Evaluation: " << this->timeEvaluation << " seconds" << std::endl;

}

void DetectionsEvaluator::evaluateSample(Sample gt, Sample detection, bool isIouTypeBbox) {

    Eval::EvalMatrix sampleEvalMatrix;

    StatsUtils::computeIOUMatrix(gt, detection, sampleEvalMatrix, isIouTypeBbox);


    std::string sampleID = gt.getSampleID();

    std::map<double, std::map<int, int>> matchingMap;
    std::map<double, std::map<double, double>> prMap;
    std::map<std::string, int> gtRegionsClassWiseCount;

    if (isIouTypeBbox) {

        auto gtRegions = gt.getRectRegions()->getRegions();
        auto detectionRegions = detection.getRectRegions()->getRegions();


        for (int i = 0; i < 10; i++) {
            std::string current_class, previous_class;
            int count = 0;
            std::map<int, bool> gtIsCrowd;
            for (auto itGt = gtRegions.begin(); itGt != gtRegions.end(); itGt++ ) {

                if (!itGt->isCrowd) {
                    this->sampleStats[this->iouThrs[i]].addGroundTruth(itGt->classID, true);
                    gtIsCrowd[itGt->uniqObjectID] = false;

                } else {
                    gtIsCrowd[itGt->uniqObjectID] = true;
                    this->sampleStats[this->iouThrs[i]].addGroundTruth(itGt->classID, false);
                }

            }

            for (auto itDetection = detectionRegions.begin(); itDetection != detectionRegions.end(); itDetection++) {
                previous_class = current_class;
                current_class = itDetection->classID;

                if (!previous_class.empty()) {
                    if (current_class == previous_class) {
                        count++;
                    } else {
                        count = 0;
                    }
                }
                if (sampleEvalMatrix[current_class].empty()) {
                    LOG(INFO) << "IOU Matrix for " << sampleID << " and class " << current_class << " is empty for this Detection Ground Truth Pair" << '\n';

                }


                double iou = std::min(this->iouThrs[i],1-1e-10);
                int m = -1;
                bool isCrowd_local;
                std::string current_class_gt, previous_class_gt;
                int count2 = 0;

                for (auto itGt = gtRegions.begin(); itGt != gtRegions.end(); itGt++) {

                    previous_class_gt = current_class_gt;
                    current_class_gt = itGt->classID;
                    if (current_class_gt != current_class)
                        continue;

                    if (!previous_class_gt.empty()) {
                        if (current_class_gt == previous_class_gt) {
                            count2++;
                        } else {
                            count2 = 0;
                        }
                    }

                    if (matchingMap[this->iouThrs[i]].find(itGt->uniqObjectID) != matchingMap[this->iouThrs[i]].end() && !itGt->isCrowd) {
                        continue;
                    }

                    if (m >-1 && !gtIsCrowd[m] && itGt->isCrowd)
                       break;

                    if (sampleEvalMatrix[current_class][count][count2] < iou)
                        continue;

                    iou=sampleEvalMatrix[current_class][count][count2];
                    m=itGt->uniqObjectID;

                }


                if (m ==-1) {
                    this->sampleStats[this->iouThrs[i]].addFalsePositive(itDetection->classID, itDetection->confidence_score);
                    continue;
                }

                if (gtIsCrowd[m]) {
                    this->sampleStats[this->iouThrs[i]].addIgnore(itDetection->classID, itDetection->confidence_score);
                    continue;
                }

                matchingMap[this->iouThrs[i]][m] = itDetection->uniqObjectID;


                this->sampleStats[this->iouThrs[i]].addTruePositive(itDetection->classID, itDetection->confidence_score);

            }
        }


    } else {

        auto gtRegions = gt.getRleRegions()->getRegions();
        auto detectionRegions = detection.getRleRegions()->getRegions();


        for (int i = 0; i < 10; i++) {
            std::string current_class, previous_class;
            int count = 0;
            std::map<int, bool> gtIsCrowd;
            for (auto itGt = gtRegions.begin(); itGt != gtRegions.end(); itGt++ ) {

                if (!itGt->isCrowd) {
                    this->sampleStats[this->iouThrs[i]].addGroundTruth(itGt->classID, true);
                    gtIsCrowd[itGt->uniqObjectID] = false;

                } else {
                    gtIsCrowd[itGt->uniqObjectID] = true;
                    this->sampleStats[this->iouThrs[i]].addGroundTruth(itGt->classID, false);

                }
            }

            for (auto itDetection = detectionRegions.begin(); itDetection != detectionRegions.end(); itDetection++) {
                previous_class = current_class;
                current_class = itDetection->classID;
                if (!previous_class.empty()) {
                    if (current_class == previous_class) {
                        count++;
                    } else {
                        count = 0;
                    }
                }
                if (sampleEvalMatrix[current_class].empty()) {
                    LOG(INFO) << "IOU Matrix for " << sampleID << " and class " << current_class << " is empty for this Detection Ground Truth Pair" << '\n';

                }

                double iou = std::min(this->iouThrs[i],1-1e-10);
                int m = -1;
                bool isCrowd_local;
                std::string current_class_gt, previous_class_gt;
                int count2 = 0;

                for (auto itGt = gtRegions.begin(); itGt != gtRegions.end(); itGt++) {
                    previous_class_gt = current_class_gt;
                    current_class_gt = itGt->classID;

                    if (current_class_gt != current_class)
                        continue;

                    if (!previous_class_gt.empty()) {
                        if (current_class_gt == previous_class_gt) {
                            count2++;
                        } else {
                            count2 = 0;
                        }
                    }

                    if (matchingMap[this->iouThrs[i]].find(itGt->uniqObjectID) != matchingMap[this->iouThrs[i]].end() && !itGt->isCrowd) {
                        continue;
                    }

                    if (m >-1 && !gtIsCrowd[m] && itGt->isCrowd)
                       break;

                    if (sampleEvalMatrix[current_class][count][count2] < iou)
                        continue;

                    iou=sampleEvalMatrix[current_class][count][count2];
                    m=itGt->uniqObjectID;

                }


                if (m ==-1) {
                    this->sampleStats[this->iouThrs[i]].addFalsePositive(itDetection->classID, itDetection->confidence_score);
                    continue;
                }
                if (gtIsCrowd[m]) {
                    this->sampleStats[this->iouThrs[i]].addIgnore(itDetection->classID, itDetection->confidence_score);
                    continue;
                }

                matchingMap[this->iouThrs[i]][m] = itDetection->uniqObjectID;


                this->sampleStats[this->iouThrs[i]].addTruePositive(itDetection->classID, itDetection->confidence_score);

            }
        }



    }


}

void DetectionsEvaluator::printStats() {
    //this->stats.printStats(classesToDisplay);
}


bool DetectionsEvaluator::sameClass(const std::string class1, const std::string class2) {

    if (class1.compare(class2)==0)
        return true;
    else{
        if (std::find(validMixClass.begin(), validMixClass.end(), std::make_pair(class1,class2)) != validMixClass.end())
            return true;
        if (std::find(validMixClass.begin(), validMixClass.end(), std::make_pair(class2,class1)) != validMixClass.end())
            return true;
    }
    return false;
}


void DetectionsEvaluator::addValidMixClass(const std::string classA, const std::string classB){
    //B is valid by detecting object as A
    this->validMixClass.push_back(std::make_pair(classA,classB));
}

void DetectionsEvaluator::addClassToDisplay(const std::string &classID) {
    this->classesToDisplay.push_back(classID);
}
