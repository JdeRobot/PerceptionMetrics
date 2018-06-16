#include <Common/Sample.h>
#include <DatasetConverters/ClassTypeGeneric.h>
#include "TensorFlowInferencer.h"

TensorFlowInferencer::TensorFlowInferencer(const std::string &netConfig, const std::string &netWeights,const std::string& classNamesFile): netConfig(netConfig),netWeights(netWeights) {

	std::cout << "in tensorflow constructor" << '\n';
	this->classNamesFile=classNamesFile;

	/* Code below adds path of python models to sys.path so as to enable python
	interpreter to import custom python modules from the path mentioned. This will
	prevent adding python path manually.
	*/

	std::string file_path = __FILE__;
	std::string dir_path = file_path.substr(0, file_path.rfind("/"));
	dir_path = dir_path + "/../python_modules";

	std::string string_to_run = "import sys\nsys.path.append('" + dir_path + "')\n";

	Py_Initialize();

	PyRun_SimpleString(string_to_run.c_str());


	init();

	std::cout << "InterPreter Initailized" << '\n';

	pName = PyString_FromString("tensorflow_detect");


	pModule = PyImport_Import(pName);
	Py_DECREF(pName);

	std::cout << "Loading Detection Graph" << '\n';

	if (pModule != NULL) {
		pClass = PyObject_GetAttrString(pModule, "TensorFlowDetector");

		pArgs = PyTuple_New(1);

		pmodel = PyString_FromString(netWeights.c_str());


		/* pValue reference stolen here: */
		PyTuple_SetItem(pArgs, 0, pmodel);
		/* pFunc is a new reference */
		pInstance = PyInstance_New(pClass, pArgs, NULL);

		if (pInstance == NULL)
		{
			Py_DECREF(pArgs);
			PyErr_Print();
		}

	} else {
		if (PyErr_Occurred())
		PyErr_Print();
		fprintf(stderr, "Cannot find function \"tensorflow_detect\"\n");
	}

	std::cout << "Detection Graph Loaded" << '\n';

}

void TensorFlowInferencer::init()
{
	import_array();
}

Sample TensorFlowInferencer::detectImp(const cv::Mat &image) {

	if(PyErr_CheckSignals() == -1) {
		throw std::runtime_error("Keyboard Interrupt");
	}

	cv::Mat rgbImage;
	cv::cvtColor(image,rgbImage,CV_BGR2RGB);

	this->detections.clear();						//remove previous detections

	int result = gettfInferences(rgbImage);

	if (result == 0) {
		std::cout << "Error Occured during getting inferences" << '\n';
	}

	Sample sample;
	RectRegionsPtr regions(new RectRegions());
	ClassTypeGeneric typeConverter(classNamesFile);

	for (auto it = detections.begin(), end=detections.end(); it !=end; ++it){

		typeConverter.setId(it->classId);
		regions->add(it->boundingBox,typeConverter.getClassString());
		//std::cout<< it->boundingBox.x << " " << it->boundingBox.y << " " << it->boundingBox.height << " " << it->boundingBox.width << std::endl;
		std::cout<< typeConverter.getClassString() << ": " << it->probability << std::endl;
	}

	sample.setRectRegions(regions);
	return sample;
}

/*
This function converts the output from python scripts into a fromat compatible
DetectionSuite to read bounding boxes, classes and detection scores, which are
drawn on the image to show detections.
*/

void TensorFlowInferencer::output_result(int num_detections, int width, int height, PyObject* bounding_boxes, PyObject* detection_scores, PyObject* classIds )
{


	if( PyArray_Check(bounding_boxes) && PyArray_Check(detection_scores) && PyArray_Check(classIds) ) {


		PyArrayObject* bounding_boxes_cont = PyArray_GETCONTIGUOUS( (PyArrayObject*) bounding_boxes );

		PyArrayObject* detection_scores_cont = PyArray_GETCONTIGUOUS( (PyArrayObject*) detection_scores );

		PyArrayObject* classIds_cont = PyArray_GETCONTIGUOUS( (PyArrayObject*) classIds );


		float* bounding_box_data = (float*) bounding_boxes_cont->data; // not copying data
		float* detection_scores_data = (float*) detection_scores_cont->data;
		unsigned char* classIds_data = (unsigned char*) classIds_cont->data;


		int i;
		int boxes = 0, scores = 0, classes = 0;

		for( i=0; i<num_detections; i++ ) {



			detections.push_back(detection());
			detections[i].classId = classIds_data[classes++] - 1;  // In TensorFlow id's start from 1 whereas detectionsuite starts from 0s
			detections[i].probability = detection_scores_data[scores++];

			detections[i].boundingBox.y = bounding_box_data[boxes++] * height;

			detections[i].boundingBox.x = bounding_box_data[boxes++] * width;

			detections[i].boundingBox.height = bounding_box_data[boxes++] * height - detections[i].boundingBox.y;

			detections[i].boundingBox.width = bounding_box_data[boxes++] * width - detections[i].boundingBox.x;

		}


		// clean
		Py_XDECREF(bounding_boxes);
		Py_XDECREF(detection_scores);
		Py_XDECREF(classIds);
	}
}


/* This function gets inferences from the Python script by calling coressponding
function and the uses output_result() to convert it into a DetectionSuite C++
readble format.
*/

int TensorFlowInferencer::gettfInferences(const cv::Mat& image) {


	int i, num_detections, dims, sizes[3];

	if (image.channels() == 3) {
		dims = 3;
		sizes[0] = image.rows;
		sizes[1] = image.cols;
		sizes[2] = image.channels();

	} else if (image.channels() == 1) {
		dims = 2;
		sizes[0] = image.rows;
		sizes[1] = image.cols;
	} else {
		std::cout << "Invalid Image Passed" << '\n';
		return 0;
	}


	npy_intp _sizes[CV_MAX_DIM+1];

	for( i = 0; i < dims; i++ )
	{
		_sizes[i] = sizes[i];
	}



	PyObject* mynparr = PyArray_SimpleNewFromData(dims, _sizes, NPY_UBYTE, image.data);

	if (!mynparr) {
		Py_DECREF(pArgs);
		Py_DECREF(pModule);
		fprintf(stderr, "Cannot convert argument\n");
		return 0;
	}

	//pValue = PyObject_CallObject(pFunc, pArgs);
	pValue = PyObject_CallMethodObjArgs(pInstance, PyString_FromString("detect"), mynparr, NULL);

	Py_DECREF(pArgs);
    if (pValue != NULL) {
		num_detections = _PyInt_AsInt( PyDict_GetItem(pValue, PyString_FromString("num_detections") ) );
		printf("Num Detections: %d\n",  num_detections );
		PyObject* pBounding_boxes = PyDict_GetItem(pValue, PyString_FromString("detection_boxes") );
		PyObject* pDetection_scores = PyDict_GetItem(pValue, PyString_FromString("detection_scores") );
		PyObject* classIds = PyDict_GetItem(pValue, PyString_FromString("detection_classes") );
		output_result(num_detections, image.cols, image.rows, pBounding_boxes, pDetection_scores, classIds);
		Py_DECREF(pValue);
	}
	else {
		Py_DECREF(pClass);
		Py_DECREF(pModule);
		PyErr_Print();
		fprintf(stderr,"Call failed\n");

		return 0;
	}


	return 1;
}
