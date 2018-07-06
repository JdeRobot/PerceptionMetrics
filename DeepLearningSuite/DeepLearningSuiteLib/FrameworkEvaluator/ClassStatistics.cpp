//
// Created by frivas on 1/02/17.
//

#include <numeric>
#include "ClassStatistics.h"
#include <iostream>
#include <limits>

ClassStatistics::ClassStatistics(const std::string& classID):classID(classID),nSamples(0),truePositives(0),falsePositives(0), falseNegatives(0), trueNegatives(0){

}
ClassStatistics::ClassStatistics():classID(""),nSamples(0),truePositives(0),falsePositives(0), falseNegatives(0), trueNegatives(0){

}

double ClassStatistics::divide(double x, double y) {
   if (y == 0)
      y = std::numeric_limits<double>::min();
   return x/y;
}

double ClassStatistics::getMeanIOU() const{
   return std::accumulate( this->iou.begin(), this->iou.end(), 0.0)/this->iou.size();
}

double ClassStatistics::getAveragePrecision(std::vector<double> recallThrs) const {

   std::vector<double> pt_rc = getPrecisionForDiffRecallThrs(recallThrs);
   double precsion = 0;
   int precsionCount = 0;
   for (auto it = pt_rc.begin(); it != pt_rc.end(); it++) {
       //std::cout << *it << ' ';
       //count++;
       precsion += *it;
       precsionCount++;

   }

   return precsion/precsionCount;

}


std::vector<double> ClassStatistics::getPrecisionForDiffRecallThrs(std::vector<double> recallThrs) const{
    std::vector<double> precisionForDiffRecallThrs(recallThrs.size());
    //std::vector<double> scoresForDiffRecallThrs;

    //std::cout << recallThrs.size() << ' ';

    std::vector<double> precisionArrayOp = getPrecisionArrayOp();
    std::vector<double> recallArray = getRecallArray();


    //int recallThrsSize = sizeof(recallThrs) / sizeof(recallThrs[0]);
    //std::cout << recallThrs.size() << '\n';

    bool isEmpty = precisionArrayOp.size() == 0;

    std::vector<double>::iterator it;
    for (int i = 0; i < recallThrs.size(); i++) {

        if (!isEmpty) {
            it = std::lower_bound(recallArray.begin(), recallArray.end(), recallThrs[i]);
            int index;
            if (it != recallArray.end()) {
                index = std::distance(recallArray.begin(), it);

                //std::cout << index << '\n';
                /*if (i == 49 || i == 50 || i == 51) {
                    std::cout << " Index: " << index << ' ';
                }*/
                precisionForDiffRecallThrs[i] = precisionArrayOp[index];
            } else {
                /*if (*(--it) == recallThrs[i]) {
                    index = std::distance(recallArray.begin(), it);
                    precisionForDiffRecallThrs.push_back(precisionArrayOp[index]);
                } else {*/
                    precisionForDiffRecallThrs[i] = 0;
                //}

            }
        } else {
            precisionForDiffRecallThrs.push_back(0);
        }

        //scoresForDiffRecallThrs.push_back(*(this->confScores.begin() + index));
    }

    return precisionForDiffRecallThrs;

}

std::vector<double> ClassStatistics::getPrecisionArrayOp() const{
    std::vector<double> precision_array = getPrecisionArray();
    if (precision_array.size() == 0 || precision_array.size() == 1) {
        return precision_array;
    }

    //std::vector<double>::reverse_iterator rit;
    //int i = 0;
    for (auto it = ++(precision_array.rbegin()); it != precision_array.rend(); it++) {
        //rit = it;
        //iter++;
        //std::cout << *it << '\n';
        if (*it < *std::prev(it)) {
            *it = *std::prev(it);
        }
    }

    return precision_array;
}

std::vector<double> ClassStatistics::getPrecisionArray() const{
    //std::cout << this->truePositives.size() << '\n';
    //std::cout << this->falsePositives.size() << '\n';
    std::vector<int> cumulative_truePositives(this->truePositives.size());
    std::partial_sum(this->truePositives.begin(), this->truePositives.end(), cumulative_truePositives.begin());
    std::vector<int> cumulative_falsePositives(this->falsePositives.size());
    std::partial_sum(this->falsePositives.begin(), this->falsePositives.end(), cumulative_falsePositives.begin());
    std::vector<int> cum_sum(this->truePositives.size());
    std::transform (cumulative_truePositives.begin(), cumulative_truePositives.end(), cumulative_falsePositives.begin(), cum_sum.begin(), std::plus<int>());
    std::vector<double> result(this->truePositives.size());
    std::transform (cumulative_truePositives.begin(), cumulative_truePositives.end(), cum_sum.begin(), result.begin(), divide);

    /*for (auto it = this->truePositives.begin(); it != this->truePositives.end(); it++) {
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
    */


    //result.push_back(2.4);
    return result;
}

double ClassStatistics::getRecall() const{
   std::vector<double> recall = getRecallArray();

   return recall.empty() ? 0 : recall[recall.size() - 1];
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

        result[i] = *it / (double)this->numGroundTruthsReg;
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
