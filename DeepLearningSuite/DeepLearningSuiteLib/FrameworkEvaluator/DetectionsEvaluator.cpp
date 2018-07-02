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
        this->recallThrs[i] = 0.01*i;
    } // Initializing Recall Thersholds with 101 values starting
      // from 0 with a difference of 0.01.

}

void DetectionsEvaluator::evaluate() {
    int counter=0;
    int gtSamples = this->gt->getNumberOfElements();
    int detectionSamples = this->detections->getNumberOfElements();

    Sample testSample;
    int count = 0;
    while (this->gt->getNextSample(testSample)){
        testSample.print();
        if (count == 500) {
            break;
        }
        count++;
    }

    /*while (this->detections->getNextSample(testSample)){
        testSample.print();
    }*/

    if (gtSamples != detectionSamples){
        LOG(WARNING) << "Both dataset has not the same number of elements";
    }

    ClassTypeMapper classMapper(this->gt->getClassNamesFile());
    this->classMapping = classMapper.mapFile(this->detections->getClassNamesFile());


    Sample gtSample;
    Sample detectionSample;
    Sample* gtSamplePtr;

    Eval::EvalMatrix evalmatrix;

    while (this->detections->getNextSample(detectionSample)){
        counter++;
        std::cout << "Evaluating: " << detectionSample.getSampleID() << "(" << counter << "/" << gtSamples << ")" << std::endl;


        //this->detections->getNextSample(detectionSample);
        if(!this->gt->getSampleBySampleID(&gtSamplePtr, detectionSample.getSampleID())) {
            std::cout << "Can't Find Sample" << '\n';
            continue;
        }

        if (gtSamplePtr->getSampleID().compare(detectionSample.getSampleID()) != 0){
            const std::string error="Both dataset has not the same structure ids mismatch from:" + gtSample.getSampleID() + " to " + detectionSample.getSampleID();
            LOG(WARNING) << error;
            throw error;
        }

        detectionSample.print();
        std::cout << "Ground Truth" << '\n';
        gtSamplePtr->print();
        evaluateSample(*gtSamplePtr,detectionSample, evalmatrix);
        std::cout << "Size: " << gtSamplePtr->getColorImage().size() << '\n';
        //Eval::printMatrix(evalmatrix);
        //printStats();

        if (this->debug){
            cv::imshow("GT", gtSample.getSampledColorImage());
            Sample detectionWithImage=detectionSample;
            detectionWithImage.setColorImage(gtSample.getColorImage());
            cv::imshow("Detection", detectionWithImage.getSampledColorImage());
            cv::waitKey(100);
        }

    }

    /*for (auto itr = this->sampleStats.begin(); itr != this->sampleStats.end(); itr++) {
        std::cout << "IOU Threshold: " << itr->first << '\n';
        std::map<std::string, ClassStatistics> mystats = itr->second.getStats();
        for (auto iter = mystats.begin(); iter != mystats.end(); iter++) {
            std::cout << "ClassID: " << iter->first << '\n';
            std::vector<double> pr = iter->second.getPrecisionArray();
            std::vector<double> rc = iter->second.getRecallArray();
            //int count = 0;
            std::cout << "Num GroundTruth: " << iter->second.numGroundTruths << '\n';
            for (auto it = pr.begin(); it != pr.end(); it++) {
                std::cout << *it << ' ';
                //std::cout << rc[count] << '\n';
                //count++;
            }
            std::cout << '\n';
            for (auto it = rc.begin(); it != rc.end(); it++) {
                std::cout << *it << ' ';
                //count++;
            }
            std::cout << '\n';
        }
    }*/


    cv::destroyAllWindows();
    std::cout << "Evaluated Successfully" << '\n';
}

/*
void DetectionsEvaluator::evaluateSamples(Sample gt, Sample detection) {
    GlobalStats currentStats;

    auto detectionRegions = detection.getRectRegions()->getRegions();
    for (auto itDetection = detectionRegions.begin(), end = detectionRegions.end(); itDetection != end; ++itDetection) {
        std::string detectionClass = this->classMapping[itDetection->classID];
        if (detectionClass.empty())
            continue;

        bool matched = false;
        auto gtRegions = gt.getRectRegions()->getRegions();
        for (auto itGT = gtRegions.begin(), endGT = gtRegions.end(); itGT != endGT; ++itGT) {
            if (sameClass(detectionClass,itGT->classID )){
                double iouValue = StatsUtils::getIOU(itGT->region, itDetection->region);
                if (iouValue > thIOU) {
                    if (detectionClass.compare(itGT->classID) ==0) {
                        stats.addTruePositive(detectionClass);
                        currentStats.addTruePositive(detectionClass);
                        stats.addIOU(detectionClass, iouValue);
                        currentStats.addIOU(detectionClass, iouValue);
                    }
                    else{
                        stats.addTruePositive(itGT->classID);
                        currentStats.addTruePositive(itGT->classID);
                        stats.addIOU(itGT->classID,iouValue );
                        currentStats.addIOU(itGT->classID,iouValue );

                    }
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) {
            stats.addFalsePositive(detectionClass);
            currentStats.addFalsePositive(detectionClass);
        }

    }

    auto gtRegions = gt.getRectRegions()->getRegions();
    for (auto itGT = gtRegions.begin(), endGT = gtRegions.end(); itGT != endGT; ++itGT) {
        bool matched = false;
        for (auto itDetection = detectionRegions.begin(), end = detectionRegions.end();
             itDetection != end; ++itDetection) {
                std::string detectionClass = this->classMapping[itDetection->classID];

                if (sameClass(itGT->classID,itDetection->classID )){
                double iouValue = StatsUtils::getIOU(itGT->region, itDetection->region);

                if (iouValue > thIOU) {
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) {
            stats.addFalseNegative(itGT->classID);
            currentStats.addFalseNegative(itGT->classID);
        }
    }

}
*/
void DetectionsEvaluator::evaluateSample(Sample gt, Sample detection, Eval::EvalMatrix& evalmatrix) {

    StatsUtils::computeIOUMatrix(gt, detection, evalmatrix);


    std::string sampleID = gt.getSampleID();

    auto detectionRegions = detection.getRectRegions()->getRegions();

    auto gtRegions = gt.getRectRegions()->getRegions();

    /*for (auto itDetection = detectionRegions.begin(); itDetection != detectionRegions.end(); itDetection++) {


    }*/

    for (auto it = evalmatrix[sampleID].begin(); it != evalmatrix[sampleID].end(); it++) {
        std::cout << "ClassID: " << it->first << '\n';
        for (auto iter = it->second.begin(); iter != it->second.end(); iter++ ) {
            for (auto iterate = iter->begin(); iterate != iter->end(); iterate++) {
                std::cout << *iterate << " ";
            }
            std::cout << '\n';
        }

    }



    std::map<double, std::map<int, int>> matchingMap;
    std::map<double, std::map<double, double>> prMap;
    std::map<std::string, int> gtRegionsClassWiseCount;

    //std::map<double, GlobalStats> this->sampleStats;

    for (int i = 0; i < 10; i++) {
        std::string current_class, previous_class;
        int count = 0;
        std::map<int, bool> gtIsCrowd;
        for (auto itGt = gtRegions.begin(); itGt != gtRegions.end(); itGt++ ) {
            if (!itGt->isCrowd) {
                this->sampleStats[this->iouThrs[i]].addGroundTruth(itGt->classID);
                gtIsCrowd[itGt->uniqObjectID] = false;
                std::cout << "0" << ' ';
            } else {
                gtIsCrowd[itGt->uniqObjectID] = true;
                std::cout << "111111111111111111111111111111111111111111111111111111111111111111111111111111" << '\n';
            }
            std::cout << itGt->classID << ' ';
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
            if (evalmatrix[sampleID][current_class].empty()) {
                std::cout << "IOU Matrix for " << sampleID << " and class " << current_class << " is empty for this Detection Ground Truth Pair" << '\n';
                continue;
            }

            std::string gtClass = this->classMapping[current_class];


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
                       // if this gt already matched, and not a crowd, continue
                if (matchingMap.find(itGt->uniqObjectID) != matchingMap.end() && !itGt->isCrowd)
                    continue;
                       //if gtm[tind,gind]>0 and not iscrowd[gind]:
                    //       continue
                       //# if dt matched to reg gt, and on ignore gt, stop
                if (m >-1 && !gtIsCrowd[m] && itGt->isCrowd)
                   break;
                       // continue to next gt unless better match made
                if (evalmatrix[sampleID][current_class][count][count2] < iou)
                    continue;
                       //# if match successful and best so far, store appropriately
                iou=evalmatrix[sampleID][current_class][count][count2];
                m=itGt->uniqObjectID;
                   //# if match made store id of match for both dt and gt
            }

            if (gtIsCrowd[m]) {
                std::cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" << '\n';
                this->sampleStats[this->iouThrs[i]].addIgnore(itDetection->classID, itDetection->confidence_score);
                continue;
            }
            if (m ==-1) {
                this->sampleStats[this->iouThrs[i]].addFalsePositive(itDetection->classID, itDetection->confidence_score);
                continue;
            }

            matchingMap[this->iouThrs[i]][m] = itDetection->uniqObjectID;


            this->sampleStats[this->iouThrs[i]].addTruePositive(itDetection->classID, itDetection->confidence_score);

            /*dtIg[tind,dind] = gtIg[m]
            dtm[tind,dind]  = gt[m]['id']
            gtm[tind,m]     = d['id']
            */
        }
    }

    std::cout << sampleID << '\n';

    /*for (auto itr = matchingMap.begin(); itr != matchingMap.end(); itr++) {
        std::cout << "For IOU: " << itr->first << '\n';
        for (auto it = itr->second.begin(); it != itr->second.end(); it++) {
            //std::cout << "ID: " << it->first << '\n';
            std::cout << it->first << " " << it->second << '\n';
                    //}
        }

    }*/



}

void DetectionsEvaluator::printStats() {
    //this->stats.printStats(classesToDisplay);
}

GlobalStats DetectionsEvaluator::getStats() {
    return this->stats;
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
