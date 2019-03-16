#include <Common/Sample.h>
#include <DatasetConverters/ClassTypeGeneric.h>
#include "KerasInferencer.h"
#include <glog/logging.h>
KerasInferencer::KerasInferencer(const std::string &netConfig, const std::string &netWeights,const std::string& classNamesFile): netConfig(netConfig),netWeights(netWeights) {

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

	LOG(INFO) << "InterPreter Initailized" << '\n';

	pName = PyString_FromString("keras_detect");


	pModule = PyImport_Import(pName);
	Py_DECREF(pName);

	LOG(INFO) << "Loading Keras Model" << '\n';

	if (pModule != NULL) {
		pClass = PyObject_GetAttrString(pModule, "KerasDetector");

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
		fprintf(stderr, "Cannot find function \"keras_detect\"\n");
	}

	LOG(INFO) << "Loaded Keras Model" << '\n';

}

void KerasInferencer::init()
{
	import_array();
}

Sample KerasInferencer::detectImp(const cv::Mat &image, double confidence_threshold) {

	if(PyErr_CheckSignals() == -1) {
		throw std::runtime_error("Keyboard Interrupt");
	}

	cv::Mat rgbImage;
	cv::cvtColor(image,rgbImage,cv::COLOR_BGR2RGB);

	this->detections.clear();						//remove previous detections

	int result = getKerasInferences(rgbImage, confidence_threshold);

	if (result == 0) {
		LOG(ERROR) << "Error Occured during getting inferences" << '\n';
	}

	Sample sample;
	RectRegionsPtr regions(new RectRegions());
	ClassTypeGeneric typeConverter(classNamesFile);

	for (auto it = detections.begin(), end=detections.end(); it !=end; ++it){

		typeConverter.setId(it->classId);
		regions->add(it->boundingBox,typeConverter.getClassString(), it->probability);
		//std::cout<< it->boundingBox.x << " " << it->boundingBox.y << " " << it->boundingBox.height << " " << it->boundingBox.width << std::endl;
		LOG(INFO)<< typeConverter.getClassString() << ": " << it->probability << std::endl;
	}

	sample.setColorImage(image);
	sample.setRectRegions(regions);
	return sample;
}

/*
This function converts the output from python scripts into a fromat compatible
DetectionSuite to read bounding boxes, classes and detection scores, which are
drawn on the image to show detections.
*/

void KerasInferencer::output_result(PyObject* result, int sizes[])
{

    int* dims;

	if( PyArray_Check(result)) {


		PyArrayObject* result_cont = PyArray_GETCONTIGUOUS( (PyArrayObject*) result );

		float* result_data = (float*) result_cont->data; // not copying data

        dims = (int*) PyArray_SHAPE(result_cont);

		int i;
		int k = 0;


		for( i=0; i<dims[0]; i++ ) {

			detections.push_back(detection());
			detections[i].classId = (int) result_data[k++] - 1;  // In Keras id's start from 1 whereas detectionsuite starts from 0s
			detections[i].probability = result_data[k++];

			detections[i].boundingBox.x = result_data[k++] * sizes[1];

			detections[i].boundingBox.y = result_data[k++] * sizes[0];

			detections[i].boundingBox.width = result_data[k++] * sizes[1] - detections[i].boundingBox.x;

			detections[i].boundingBox.height = result_data[k++] * sizes[0] - detections[i].boundingBox.y;



		}


		// clean
		Py_XDECREF(result_cont);
	}
}


/* This function gets inferences from the Python script by calling coressponding
function and the uses output_result() to convert it into a DetectionSuite C++
readble format.
*/

int KerasInferencer::getKerasInferences(const cv::Mat& image, double confidence_threshold) {


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
		LOG(ERROR) << "Invalid Image Passed" << '\n';
		return 0;
	}


	npy_intp _sizes[4];

	for( i = 0; i < dims; i++ )
	{
		_sizes[i] = sizes[i];
	}



	PyObject* mynparr = PyArray_SimpleNewFromData(dims, _sizes, NPY_UBYTE, image.data);
	PyObject* conf = PyFloat_FromDouble(confidence_threshold);

	if (!mynparr || !conf) {
		Py_DECREF(pArgs);
		Py_DECREF(pModule);
		fprintf(stderr, "Cannot convert argument\n");
		return 0;
	}

	//pValue = PyObject_CallObject(pFunc, pArgs);
	pValue = PyObject_CallMethodObjArgs(pInstance, PyString_FromString("detect"), mynparr, conf, NULL);

	Py_DECREF(pArgs);
    if (pValue != NULL) {
		output_result(pValue, sizes);
        LOG(INFO) << "Num Detections: " << this->detections.size() << '\n';
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
