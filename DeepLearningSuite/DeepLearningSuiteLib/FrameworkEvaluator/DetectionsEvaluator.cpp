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

void DetectionsEvaluator::accumulateResults() {

    int start_s=clock();

    unsigned int index = 0;
    for (auto itr = this->sampleStats.begin(); itr != this->sampleStats.end(); itr++) {

        //std::cout << "IOU Threshold: " << itr->first << '\n';
        std::map<std::string, ClassStatistics> mystats = itr->second.getStats();
        //double ap = 0;
        //int size = 0;
        int totalCount = 0;
        double totalPrecision = 0;
        double totalRecall = 0;

        int classCount = 0;

        for (auto iter = mystats.begin(); iter != mystats.end(); iter++) {
            //std::cout << "ClassID: " << iter->first << '\n';
            if (iter->second.numGroundTruthsReg == 0)
                continue;
            std::vector<double> pr = iter->second.getPrecisionArray();

            std::vector<double> rc = iter->second.getRecallArray();
            //int count = 0;
            //std::cout << "Num GroundTruth: " << iter->second.numGroundTruthsReg << '\n';
            /*for (auto it = pr.begin(); it != pr.end(); it++) {
                std::cout << *it << ' ';
                //std::cout << rc[count] << '\n';
                //count++;
            }
            std::cout << '\n';
            std::vector<double> pr_op = iter->second.getPrecisionArrayOp();
            for (auto it = pr_op.begin(); it != pr_op.end(); it++) {
                std::cout << *it << ' ';
                //count++;
            }*/
            //std::cout << '\n';

            double recall = 0;
            /*std::cout << "Priniting Recall Array" << '\n';
            for (auto it = rc.begin(); it != rc.end(); it++) {
                std::cout << *it << ' ';
                //count++;
                recallCount++;
                recall += *it;
            }*/

            recall = iter->second.getRecall();
            //std::cout << "Printing Recall Value: " << recall << '\n';
            this->classWiseMeanAR[iter->first] += recall / 10;            // 10 IOU Thresholds,
                                                                    // mean will be calculated directly
            totalRecall += recall;

            //std::cout << totalRecall << " " << recall << " " << recallCount << '\n';

            //std::cout << '\n';
            /*std::vector<double> pt_rc = iter->second.getPrecisionForDiffRecallThrs(this->recallThrs);
            double precsion = 0;
            int precsionCount = 0;
            for (auto it = pt_rc.begin(); it != pt_rc.end(); it++) {
                //std::cout << *it << ' ';
                //count++;
                precsion += *it;
                precsionCount++;

            }*/

            double precision = iter->second.getAveragePrecision(this->recallThrs);
            this->classWiseMeanAP[iter->first] += precision / 10;     // 10 IOU Thresholds,
                                                                // mean will be calculated directly
            totalPrecision += precision;

            totalCount++;
            //std::cout << average/count;
            //std::cout << '\n';

        }
        //std::cout << ap << " " << size << " " << totalCount << " " << totalAverage << '\n';
        this->ApDiffIou[index] = totalPrecision / totalCount;
        this->ArDiffIou[index] = totalRecall / totalCount;
        //std::cout << "Recall Count: " << totalRecallCount << '\n';
        ++index;
    }

    int stop_s=clock();
    std::cout << "Time Taken in Accumulation: " << (stop_s-start_s)/double(CLOCKS_PER_SEC) << " seconds" << std::endl;
    std::cout << std::fixed;
    std::cout << std::setprecision(8);


    for (int i = 0; i < this->ApDiffIou.size(); i++) {
        std::cout << "AP for IOU " << this->iouThrs[i] <<  ": \t" << this->ApDiffIou[i] << '\n';
    }

    for (int i = 0; i < this->ArDiffIou.size(); i++) {
        std::cout << "AR for IOU " << this->iouThrs[i] <<  ": \t" << this->ArDiffIou[i] << '\n';
    }

    std::cout << "AP for IOU 0.5:0.95 \t" << this->ApDiffIou.sum()/10 << '\n';

    std::cout << "AR for IOU 0.5:0.95 \t" << this->ArDiffIou.sum()/10 << '\n';

    cv::destroyAllWindows();

    std::cout << "Evaluated Successfully" << '\n';

}

void DetectionsEvaluator::evaluate() {
    int counter=-1;
    int gtSamples = this->gt->getNumberOfElements();
    int detectionSamples = this->detections->getNumberOfElements();


    int start_s=clock();
	// the code you wish to time goes here

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
    //Sample* gtSamplePtr;



    /*int imgIdsToEval[100] = {42, 73, 74, 133, 136, 139, 143, 164, 192, 196, 208, 241, 257, 283, 285, 294, 328, 338,
                        357, 359, 360, 387, 395, 397, 400, 415, 428, 459, 472, 474, 486, 488, 502, 520, 536, 544,
                        564, 569, 589, 590, 599, 623, 626, 632, 636, 641, 661, 675, 692, 693, 699, 711, 715, 724,
                         730, 757, 761, 764, 772, 775, 776, 785, 802, 810, 827, 831, 836, 872, 873, 885, 923, 939,
                         962, 969, 974, 985, 987, 999, 1000, 1029, 1063, 1064, 1083, 1089, 1103, 1138, 1146, 1149,
                          1153, 1164, 1171, 1176, 1180, 1205, 1228, 1244, 1268, 1270, 1290, 1292};

	*/

    while (this->gt->getNextSample(gtSample)) {
        counter++;
       // if (!this->gt->getSampleBySampleID(&gtSamplePtr, imgIdsToEval[counter])){
        //    continue;
        //}


        //Sample* detectionSample;
        this->detections->getNextSample(detectionSample);
        /*if(!this->detections->getSampleBySampleID(&detectionSample, gtSamplePtr->getSampleID())) {
            std::cout << "Can't Find Sample, creating dummy sample i.e, assuming that the object detector didn't detect any objects in this image" << '\n';
            //continue;
            detectionSample = new Sample();
            detectionSample->setSampleID(gtSamplePtr->getSampleID());
        }*/

        std::cout << "Evaluating: " << detectionSample.getSampleID() << "(" << counter << "/" << gtSamples << ")" << std::endl;


        if (gtSample.getSampleID().compare(detectionSample.getSampleID()) != 0){
            LOG(WARNING) << "No detection sample available, Creating Dummy Sample\n";
            Sample dummy;
            dummy.setSampleID(gtSample.getSampleID());
            dummy.setColorImage(gtSample.getColorImagePath());
            evaluateSample(gtSample, dummy);
            this->detections->decrementReaderCounter();
            const std::string error="Both dataset has not the same structure ids mismatch from:" + gtSample.getSampleID() + " to " + detectionSample.getSampleID();
            LOG(WARNING) << error;
            //throw error;
            //detectionSample = *dummy;
        } else {
            evaluateSample(gtSample,detectionSample);
        }

        //detectionSample->print();
        //std::cout << "Ground Truth" << '\n';
        //gtSamplePtr->print();

        //std::cout << "Size: " << gtSamplePtr->getColorImage().size() << '\n';
        //Eval::printMatrix(evalmatrix);
        //printStats();

        /*if (this->debug){
            cv::imshow("GT", gtSample.getSampledColorImage());
            cv::imshow("Detection", detectionSample.getSampledColorImage());
            cv::waitKey(10);
        }*/

    }
    int stop_s=clock();
    std::cout << "Time Taken in Evaluation: " << (stop_s-start_s)/double(CLOCKS_PER_SEC) << " seconds" << std::endl;

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
void DetectionsEvaluator::evaluateSample(Sample gt, Sample detection) {

    Eval::EvalMatrix sampleEvalMatrix;

    StatsUtils::computeIOUMatrix(gt, detection, sampleEvalMatrix);


    std::string sampleID = gt.getSampleID();

    auto detectionRegions = detection.getRectRegions()->getRegions();

    auto gtRegions = gt.getRectRegions()->getRegions();

    /*for (auto itDetection = detectionRegions.begin(); itDetection != detectionRegions.end(); itDetection++) {


    }*/

    /*for (auto it = sampleEvalMatrix.begin(); it != sampleEvalMatrix.end(); it++) {
        std::cout << "ClassID: " << it->first << '\n';
        for (auto iter = it->second.begin(); iter != it->second.end(); iter++ ) {
            for (auto iterate = iter->begin(); iterate != iter->end(); iterate++) {
                std::cout << *iterate << " ";
            }
            std::cout << '\n';
        }

    }*/



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
                this->sampleStats[this->iouThrs[i]].addGroundTruth(itGt->classID, true);
                gtIsCrowd[itGt->uniqObjectID] = false;
                //std::cout << "0" << ' ';
            } else {
                gtIsCrowd[itGt->uniqObjectID] = true;
                this->sampleStats[this->iouThrs[i]].addGroundTruth(itGt->classID, false);
                //std::cout << "111111111111111111111111111111111111111111111111111111111111111111111111111111" << '\n';
            }
            //std::cout << itGt->classID << ' ';
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
                std::cout << "IOU Matrix for " << sampleID << " and class " << current_class << " is empty for this Detection Ground Truth Pair" << '\n';
                // This is also a false positive
                //continue;
            }

            //std::cout << itDetection->classID << " " << itDetection->confidence_score <<  '\n';


            //std::string gtClass = this->classMapping[current_class];


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
                //std::cout << itGt->isCrowd << " " << itGt->uniqObjectID << '\n';
                       // if this gt already matched, and not a crowd, continue
                if (matchingMap[this->iouThrs[i]].find(itGt->uniqObjectID) != matchingMap[this->iouThrs[i]].end() && !itGt->isCrowd) {
                    //std::cout << "to continue " << itGt->uniqObjectID << '\n';
                    continue;
                }
                       //if gtm[tind,gind]>0 and not iscrowd[gind]:
                    //       continue
                    //std::cout << "came here" << '\n';
                       //# if dt matched to reg gt, and on ignore gt, stop
                if (m >-1 && !gtIsCrowd[m] && itGt->isCrowd)
                   break;
                       // continue to next gt unless better match made
                if (sampleEvalMatrix[current_class][count][count2] < iou)
                    continue;
                       //# if match successful and best so far, store appropriately
                //std::cout << "came here too " << itGt->uniqObjectID << '\n';
                iou=sampleEvalMatrix[current_class][count][count2];
                m=itGt->uniqObjectID;
                   //# if match made store id of match for both dt and gt
            }

            //std::cout << itDetection->classID << " " << itDetection->confidence_score << ' ' << itDetection->uniqObjectID << ' ' <<  m  <<'\n';

            if (m ==-1) {
                this->sampleStats[this->iouThrs[i]].addFalsePositive(itDetection->classID, itDetection->confidence_score);
                continue;
            }
            if (gtIsCrowd[m]) {
                //std::cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" << '\n';
                this->sampleStats[this->iouThrs[i]].addIgnore(itDetection->classID, itDetection->confidence_score);
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

    //std::cout << sampleID << '\n';

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
