//
// Created by frivas on 1/02/17.
//

#include <numeric>
#include "ClassStatistics.h"
#include <iostream>

ClassStatistics::ClassStatistics(const std::string& classID):classID(classID),nSamples(0),truePositives(0),falsePositives(0), falseNegatives(0), trueNegatives(0){

}
ClassStatistics::ClassStatistics():classID(""),nSamples(0),truePositives(0),falsePositives(0), falseNegatives(0), trueNegatives(0){

}


double ClassStatistics::getMeanIOU() const{
   return std::accumulate( this->iou.begin(), this->iou.end(), 0.0)/this->iou.size();
}

std::vector<double> ClassStatistics::getPrecisionArray() const{
    std::cout << this->truePositives.size() << '\n';
    std::cout << this->falsePositives.size() << '\n';
    std::vector<int> cumulative_truePositives(this->truePositives.size());
    std::partial_sum(this->truePositives.begin(), this->truePositives.end(), cumulative_truePositives.begin());
    std::vector<int> cumulative_falsePositives(this->falsePositives.size());
    std::partial_sum(this->falsePositives.begin(), this->falsePositives.end(), cumulative_falsePositives.begin());
    std::vector<int> cum_sum(this->truePositives.size());
    std::transform (cumulative_truePositives.begin(), cumulative_truePositives.end(), cumulative_falsePositives.begin(), cum_sum.begin(), std::plus<int>());
    std::vector<double> result(this->truePositives.size());
    std::transform (cumulative_truePositives.begin(), cumulative_truePositives.end(), cum_sum.begin(), result.begin(), std::divides<double>());

    for (auto it = this->truePositives.begin(); it != this->truePositives.end(); it++) {
        std::cout << *it << ' ';
    }
    std::cout << '\n';
    for (auto it = this->falsePositives.begin(); it != this->falsePositives.end(); it++) {
        std::cout << *it << ' ';
    }
    std::cout << '\n';
    for (auto it = cumulative_truePositives.begin(); it != cumulative_truePositives.end(); it++) {
        std::cout << *it << ' ';
    }
    std::cout << '\n';
    for (auto it = cumulative_falsePositives.begin(); it != cumulative_falsePositives.end(); it++) {
        std::cout << *it << ' ';
    }
    std::cout << '\n';



    //result.push_back(2.4);
    return result;
}

std::vector<double> ClassStatistics::getRecallArray() const{
    std::vector<int> cumulative_truePositives(this->truePositives.size());
    std::partial_sum(this->truePositives.begin(), this->truePositives.end(), cumulative_truePositives.begin());

    std::vector<double> result(this->truePositives.size());
    //std::transform (cumulative_truePositives.begin(), cumulative_truePositives.end(), (double)this->numGroundTruths, result.begin(), std::divides<double>());
    //std::transform (cumulative_truePositives.begin(), cumulative_truePositives.end(), result.begin(),
    //           std::bind(std::divides<double>(), std::placeholders::_1, this->numGroundTruths));
    //std::cout << t << '\n';
    int i =0;
    for (auto it = cumulative_truePositives.begin(); it != cumulative_truePositives.end(); it++) {

        result[i] = *it / (double)this->numGroundTruths;
        i++;
    }

    return result;
}

/*void ClassStatistics::printStats() const{
    std::cout << "------------------------------" << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << "  Class id: " << classID << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << "TP: " << this->truePositives  << std::endl;
    std::cout << "FP: " << this->falsePositives << std::endl;
    std::cout << "FN: " << this->falseNegatives << std::endl;
    std::cout << "Mean IOU: " << getMeanIOU() << std::endl;
    std::cout << "Precision: " << getPrecision() << std::endl;
    std::cout << "Recall: " << getRecall() << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << "------------------------------" << std::endl;
}*/
