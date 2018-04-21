//#include <boost/shared_ptr.hpp>
#include "FrameworkInferencer.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>

class TensorFlowInferencer: public FrameworkInferencer {
public:
    TensorFlowInferencer(const std::string& netConfig, const std::string& netWeights, const std::string& classNamesFile);
    Sample detectImp(const cv::Mat& image);
    int gettfInferences(const cv::Mat& image);
    void output_result(int num_detections, int width, int height, PyObject* bounding_boxes, PyObject* detection_scores, PyObject* classIds );
    static void init();
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


typedef boost::shared_ptr<TensorFlowInferencer> TensorFlowInferencerPtr;
