//#include <boost/shared_ptr.hpp>
#include "FrameworkInferencer.h"
#include <boost/python.hpp>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>

class TensorFlowInferencer: public FrameworkInferencer {
public:
    TensorFlowInferencer(const std::string& netConfig, const std::string& netWeights, const std::string& classNamesFile);
    Sample detectImp(const cv::Mat& image, double confidence_threshold);
    int gettfInferences(const cv::Mat& image, double confidence_threshold);
    void output_result(int num_detections, int width, int height, PyObject* bounding_boxes, PyObject* detection_scores, PyObject* classIds, PyObject* detections_masks=NULL );
    static void init();
    static void CallBackFunc(int event, int x, int y, int flags, void* userdata);
private:
    std::string netConfig;
    std::string netWeights;
    struct detection {
        cv::Rect boundingBox;
        RLE rleRegion;
        float probability;
        int classId;
    };
    bool mousy;
    PyObject *pName, *pModule, *pClass, *pInstance;
    PyObject *pArgs, *pValue, *pmodel;

    std::vector<detection> detections;
    bool hasMasks;

};


typedef boost::shared_ptr<TensorFlowInferencer> TensorFlowInferencerPtr;
