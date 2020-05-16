#include "EvalMatrix.h"
#include <iostream>
#include <glog/logging.h>
void Eval::printMatrix(Eval::EvalMatrix matrix) {

    LOG(INFO) << "Printing Matrix" << '\n';

    for (auto itr = matrix.begin(); itr != matrix.end(); itr++) {
        LOG(INFO) << "ClassID: " << itr->first <<'\n';
        for (auto iter = itr->second.begin(); iter != itr->second.end(); iter++ ) {
            for (auto iterate = iter->begin(); iterate != iter->end(); iterate++) {
                LOG(INFO) << *iterate << " ";
            }
            LOG(INFO) << '\n';
        }
    }


}
