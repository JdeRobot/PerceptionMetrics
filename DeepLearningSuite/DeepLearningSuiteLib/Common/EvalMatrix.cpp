#include "EvalMatrix.h"
#include <iostream>

void Eval::printMatrix(Eval::EvalMatrix matrix) {

    std::cout << "Printing Matrix" << '\n';

    for (auto itr = matrix.begin(); itr != matrix.end(); itr++) {
        std::cout << "ClassID: " << itr->first <<'\n';
        for (auto iter = itr->second.begin(); iter != itr->second.end(); iter++ ) {
            for (auto iterate = iter->begin(); iterate != iter->end(); iterate++) {
                std::cout << *iterate << " ";
            }
            std::cout << '\n';
        }
    }


}
