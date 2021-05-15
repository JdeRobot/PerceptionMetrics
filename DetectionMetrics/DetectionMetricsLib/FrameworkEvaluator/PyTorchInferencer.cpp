#include <Common/Sample.h>
#include <DatasetConverters/ClassTypeGeneric.h>
#include "PyTorchInferencer.h"
#include <glog/logging.h>

void PyTorchInferencer::CallBackFunc(int event, int x, int y, int flags, void* userdata){
	((PyTorchInferencer *)(userdata))->mousy = true;
	for(auto itr = ((PyTorchInferencer *)(userdata))->detections.begin(); itr != ((PyTorchInferencer *)(userdata))->detections.end() ; itr++){
		itr->boundingBox.x = x;
		itr->boundingBox.y = y;
	}
}



PyTorchInferencer::PyTorchInferencer( const std::string &netConfig, const std::string &netWeights,const std::string& classNamesFile): netConfig(netConfig),netWeights(netWeights) {
	LOG(INFO) << "PyTorch Constructor" << '\n';
	this->classNamesFile=classNamesFile;
	this->mousy = false;
	/*
	 * Code below adds path of python models to sys.path so as to enable python
	 * interpreter to import custom python modules from the path mentioned. This will
 	 * prevent adding python path manually.
	*/

	std::string file_path = __FILE__;
	std::string dir_path = file_path.substr(0, file_path.rfind("/"));
	dir_path = dir_path + "/../python_modules";
	std::string string_to_run = "import sys\nsys.path.append('" + dir_path + "')\n";

	/* Initialize the python interpreter.Neccesary step to later call
	 * the python interpreter from any part of the application.
	*/
	
	Py_Initialize();
	PyRun_SimpleString(string_to_run.c_str());
	init();
	LOG(INFO) << "Interpreter Initialized" << '\n';
	pName = PyUnicode_FromString("pytorch_detect");
	pModule = PyImport_Import(pName);
	Py_DECREF(pName);

	LOG(INFO) << "Loading Detection Graph" << '\n';
	if (pModule != NULL) {
		pClass = PyObject_GetAttrString(pModule, "PyTorchDetector");
		pArgs = PyTuple_New(2);
		pmodel = PyUnicode_FromString(netWeights.c_str());
		pconfig = PyUnicode_FromString(netConfig.c_str());
		
		PyTuple_SetItem(pArgs, 0, pmodel);
		PyTuple_SetItem(pArgs, 1, pconfig);
		pInstance = PyObject_CallObject(pClass, pArgs);
		if (pInstance == NULL) {
			Py_DECREF(pArgs);
			PyErr_Print();
		}
	} else {
		if (PyErr_Occurred())
		PyErr_Print();
		fprintf(stderr, "Cannot find function \"pytorch_detect\"\n");
	}
	LOG(INFO) << "Detection Graph Loaded" << '\n';
}

#if PY_MAJOR_VERSION >= 3
int*
#else
void
#endif
PyTorchInferencer::init() {
	import_array();
}

Sample PyTorchInferencer::detectImp(const cv::Mat &image, double confidence_threshold) {
	LOG(ERROR) << "DETECT IMP" << "\n";
	if(PyErr_CheckSignals() == -1) {
		throw std::runtime_error("Keyboard Interrupt");
	}
	cv::Mat rgbImage;
	cv::cvtColor(image,rgbImage,cv::COLOR_BGR2RGB);
	if(!this->mousy){
		LOG(ERROR) << "DETECT IMP 2" << "\n";
		this->detections.clear();
		int result = gettfInferences(rgbImage, confidence_threshold);
		if (result == 0) {
			LOG(ERROR) << "Error Occured during getting inferences" << '\n';
		}
	}
	Sample sample;
	RectRegionsPtr regions(new RectRegions());
  	RleRegionsPtr rleRegions(new RleRegions());
	ClassTypeGeneric typeConverter(classNamesFile);
	for (auto it = detections.begin(), end=detections.end(); it !=end; ++it){
		typeConverter.setId(it->classId);
		regions->add(it->boundingBox,typeConverter.getClassString(),it->probability);
		if (this->hasMasks)
			rleRegions->add(it->rleRegion, typeConverter.getClassString(), it->probability);

		LOG(INFO)<< typeConverter.getClassString() << ": " << it->probability << std::endl;
	}


	sample.setColorImage(image);
	sample.setRectRegions(regions);
	sample.setRleRegions(rleRegions);
	sample.SetMousy(this->mousy);
	this->mousy=false;
	return sample;
}

void PyTorchInferencer::output_result(int num_detections, int width, int height, PyObject* bounding_boxes, PyObject* detection_scores, PyObject* classIds, PyObject* pDetection_masks ) {
	this->hasMasks = false;
    	int mask_dims;
    	long long int* mask_shape;
	if( PyArray_Check(bounding_boxes) && PyArray_Check(detection_scores) && PyArray_Check(classIds) ) {
		PyArrayObject* detection_masks_cont = NULL;

        	if (pDetection_masks != NULL && PyArray_Check(pDetection_masks)) {
            		detection_masks_cont = PyArray_GETCONTIGUOUS( (PyArrayObject*) pDetection_masks );
            		this->hasMasks = true;
            		mask_dims = PyArray_NDIM(detection_masks_cont);
            		if (mask_dims != 3) {
                	throw std::invalid_argument("Returned Mask by pytorch doesn't have 2 dimensions");
            		}
            		mask_shape = (long long int*) PyArray_SHAPE(detection_masks_cont);
        	}
		PyArrayObject* bounding_boxes_cont = PyArray_GETCONTIGUOUS( (PyArrayObject*) bounding_boxes );
		PyArrayObject* detection_scores_cont = PyArray_GETCONTIGUOUS( (PyArrayObject*) detection_scores );
		PyArrayObject* classIds_cont = PyArray_GETCONTIGUOUS( (PyArrayObject*) classIds );
		float* bounding_box_data = (float*) bounding_boxes_cont->data;
		float* detection_scores_data = (float*) detection_scores_cont->data;
		unsigned char* classIds_data = (unsigned char*) classIds_cont->data;
		float* detection_masks_data;
		if (this->hasMasks) {
            		detection_masks_data = (float*) detection_masks_cont->data;
        	}
		int i;
		int boxes = 0, scores = 0, classes = 0, masks = 0;

		for( i=0; i<num_detections; i++ ) {
			detections.push_back(detection());
			detections[i].classId = classIds_data[classes++] - 1;
			detections[i].probability = detection_scores_data[scores++];
			detections[i].boundingBox.x = bounding_box_data[boxes++] * width;
			detections[i].boundingBox.y = bounding_box_data[boxes++] * height;
			detections[i].boundingBox.width = bounding_box_data[boxes++] * width - detections[i].boundingBox.x;
			detections[i].boundingBox.height = bounding_box_data[boxes++] * height - detections[i].boundingBox.y;
			if (this->hasMasks) {
				cv::Mat image_mask(height, width, CV_8UC1, cv::Scalar(0));
				cv::Mat mask = cv::Mat(mask_shape[1], mask_shape[2], CV_32F, detection_masks_data + i*mask_shape[1]*mask_shape[2]);
				cv::Mat mask_r;
				cv::resize(mask, mask_r, cv::Size(detections[i].boundingBox.width, detections[i].boundingBox.height));
                		cv::Mat mask_char;
                		mask_r.convertTo(mask_char, CV_8U, 255);
                		cv::threshold(mask_char, mask_char, 127, 255, cv::THRESH_BINARY);
				mask_char.copyTo(image_mask(cv::Rect(detections[i].boundingBox.x,detections[i].boundingBox.y,detections[i].boundingBox.width, detections[i].boundingBox.height)));
				RLE forMask;
				cv::Mat t_mask = image_mask.t();
				rleEncode( &forMask, t_mask.data, t_mask.cols, t_mask.rows, 1 );
				detections[i].rleRegion = forMask;
			}
		}
		Py_XDECREF(bounding_boxes);
		Py_XDECREF(detection_scores);
		Py_XDECREF(classIds);
	}
}


int PyTorchInferencer::gettfInferences(const cv::Mat& image, double confidence_threshold) {
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
	for( i = 0; i < dims; i++ ) {
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
	pValue = PyObject_CallMethodObjArgs(pInstance, PyUnicode_FromString("detect"), mynparr, conf, NULL);
	Py_DECREF(pArgs);
	if (pValue != NULL) {
		num_detections = _PyLong_AsInt( PyDict_GetItem(pValue, PyUnicode_FromString("num_detections") ) );
		printf("Num Detections: %d\n",  num_detections );
		PyObject* pBounding_boxes = PyDict_GetItem(pValue, PyUnicode_FromString("detection_boxes") );
		PyObject* pDetection_scores = PyDict_GetItem(pValue, PyUnicode_FromString("detection_scores") );
		PyObject* classIds = PyDict_GetItem(pValue, PyUnicode_FromString("detection_classes") );
		PyObject* key = PyUnicode_FromString("detection_masks");
		if (PyDict_Contains(pValue, key)) {
			PyObject* pDetection_masks = PyDict_GetItem(pValue, PyUnicode_FromString("detection_masks") );
            		output_result(num_detections, image.cols, image.rows, pBounding_boxes, pDetection_scores, classIds, pDetection_masks);
        	} else {
            		output_result(num_detections, image.cols, image.rows, pBounding_boxes, pDetection_scores, classIds);
        	}
		Py_DECREF(pValue);
	} else {
		Py_DECREF(pClass);
		Py_DECREF(pModule);
		PyErr_Print();
		fprintf(stderr,"Call failed\n");
		return 0;
	}
	return 1;
}

