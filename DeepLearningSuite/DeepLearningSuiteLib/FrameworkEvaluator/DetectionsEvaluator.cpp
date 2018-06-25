//
// Created by frivas on 1/02/17.
//

#include <glog/logging.h>
#include <Utils/StatsUtils.h>
#include "DetectionsEvaluator.h"


DetectionsEvaluator::DetectionsEvaluator(DatasetReaderPtr gt, DatasetReaderPtr detections,bool debug):
        gt(gt),detections(detections),debug(debug) {
    thIOU = 0.5;

}

void DetectionsEvaluator::evaluate() {
    int counter=0;
    int gtSamples = this->gt->getNumberOfElements();
    int detectionSamples = this->detections->getNumberOfElements();

    if (gtSamples != detectionSamples){
        LOG(WARNING) << "Both dataset has not the same number of elements";
    }

    ClassTypeMapper classMapper(this->gt->getClassNamesFile());
    this->classMapping = classMapper.mapFile(this->detections->getClassNamesFile());


    Sample gtSample;
    Sample detectionSample;

    while (this->gt->getNextSample(gtSample)){
        counter++;
        std::cout << "Evaluating: " << gtSample.getSampleID() << "(" << counter << "/" << gtSamples << ")" << std::endl;
        this->detections->getNextSample(detectionSample);
        if (gtSample.getSampleID().compare(detectionSample.getSampleID()) != 0){
            const std::string error="Both dataset has not the same structure ids mismatch from:" + gtSample.getSampleID() + " to " + detectionSample.getSampleID();
            LOG(WARNING) << error;
            throw error;
        }

        evaluateSamples(gtSample,detectionSample);
        printStats();

        if (this->debug){
            cv::imshow("GT", gtSample.getSampledColorImage());
            Sample detectionWithImage=detectionSample;
            detectionWithImage.setColorImage(gtSample.getColorImage());
            cv::imshow("Detection", detectionWithImage.getSampledColorImage());
            cv::waitKey(100);
        }

    }

    cv::destroyAllWindows();
    std::cout << "Evaluated Successfully" << '\n';
}


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
                double iouValue = StatsUtils::getIOU(itGT->region, itDetection->region, gt.getColorImage().size());
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
                double iouValue = StatsUtils::getIOU(itGT->region, itDetection->region, gt.getColorImage().size());

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

void DetectionsEvaluator::printStats() {
    this->stats.printStats(classesToDisplay);
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
