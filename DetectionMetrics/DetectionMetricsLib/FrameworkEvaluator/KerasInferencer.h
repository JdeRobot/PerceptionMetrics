#include "pythonWrap.h"
//#include <boost/shared_ptr.hpp>
#include "FrameworkInferencer.h"
#include <boost/python.hpp>
//#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>

class KerasInferencer: public FrameworkInferencer {
public:
    KerasInferencer(const std::string& netConfig, const std::string& netWeights, const std::string& classNamesFile);
    Sample detectImp(const cv::Mat& image, double confidence_threshold);
    int getKerasInferences(const cv::Mat& image, double confidence_threshold);
    void output_result(PyObject* result, int sizes[] );
    static
    #if PY_MAJOR_VERSION >= 3
    int*
    #else
    void
    #endif 
    init();
private:
    std::string netConfig;
    std::string netWeights;
    struct detection {
        cv::Rect boundingBox;
        float probability;
        int classId;
    };

    PyObject *pName, *pModule, *pClass, *pInstance;
    PyObject *pArgs, *pValue, *pmodel;

    std::vector<detection> detections;

};


typedef boost::shared_ptr<KerasInferencer> KerasInferencerPtr;
