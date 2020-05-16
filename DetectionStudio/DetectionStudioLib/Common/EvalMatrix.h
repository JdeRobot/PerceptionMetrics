#ifndef SAMPLERGENERATOR_EVALMATRIX_H
#define SAMPLERGENERATOR_EVALMATRIX_H

#include "Matrix.h"
#include <map>

namespace Eval {

using EvalMatrix = std::map<std::string, Matrix<double>>;

using DetectionsMatcher = std::map<double, std::map<std::string, std::map<int, int>>>;

void printMatrix(EvalMatrix matrix);

}
#endif //SAMPLERGENERATOR_MATRIX_H
