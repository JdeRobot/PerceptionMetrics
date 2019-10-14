#include <Common/Sample.h>
#include <DatasetConverters/ClassTypeGeneric.h>
#include "TensorFlowInferencer.h"
#include <glog/logging.h>

// Construcor function for the tensorflow Inferencer
void TensorFlowInferencer::CallBackFunc(int event, int x, int y, int flags, void* userdata){
	((TensorFlowInferencer *)(userdata))->mousy = true;
	for(auto itr = ((TensorFlowInferencer *)(userdata))->detections.begin(); itr !=((TensorFlowInferencer *)(userdata))->detections.end() ; itr++){
		itr->boundingBox.x = x;
		itr->boundingBox.y = y;
	}
	// LOG(INFO) << "This is x : " << x << std::endl;
}

TensorFlowInferencer::TensorFlowInferencer(const std::string &netConfig, const std::string &netWeights,const std::string& classNamesFile): netConfig(netConfig),netWeights(netWeights) {
	LOG(INFO) << "in tensorflow constructor" << '\n';
	this->classNamesFile=classNamesFile;
	this->mousy = false;
	/* Code below adds path of python models to sys.path so as to enable python
	interpreter to import custom python modules from the path mentioned. This will
	prevent adding python path manually.
	*/

	std::string file_path = __FILE__;
	std::string dir_path = file_path.substr(0, file_path.rfind("/"));
	dir_path = dir_path + "/../python_modules";

	std::string string_to_run = "import sys\nsys.path.append('" + dir_path + "')\n";

	/* Initialize the python interpreter.Neccesary step to later call
		 the python interpreter from any part of the application.*/
	Py_Initialize();

	/* Any python code is run in this format.
		 str = "print('Hello World')"
		 PyRun_SimpleString(str) */
	PyRun_SimpleString(string_to_run.c_str());


	init();

	LOG(INFO) << "InterPreter Initailized" << '\n';

	pName = PyString_FromString("tensorflow_detect");


	pModule = PyImport_Import(pName);
	Py_DECREF(pName);

	LOG(INFO) << "Loading Detection Graph" << '\n';

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

	LOG(INFO) << "Detection Graph Loaded" << '\n';

}

void TensorFlowInferencer::init()
{
	import_array();
}

Sample TensorFlowInferencer::detectImp(const cv::Mat &image, double confidence_threshold) {
	// cv::setMouseCallback("Detection", TensorFlowInferencer::CallBackFunc ,this);
	if(PyErr_CheckSignals() == -1) {
		throw std::runtime_error("Keyboard Interrupt");
	}

	/*
		Initialize a matrix to store the image in RGB format.
		Currently the format is BGR,below function cvtColor
		takes the image and stores the new RGB format in rgbImage.
	*/
	cv::Mat rgbImage;
	cv::cvtColor(image,rgbImage,cv::COLOR_BGR2RGB);
	if(!this->mousy){

		this->detections.clear();						//remove previous detections

		/*
			 Get the tensorflow inferences of an image/frame provided the image
			 and the confidence threshold, and store them in result.
		*/
		int result = gettfInferences(rgbImage, confidence_threshold);

		if (result == 0) {
			LOG(ERROR) << "Error Occured during getting inferences" << '\n';
		}
	}

	Sample sample;
	RectRegionsPtr regions(new RectRegions());
  RleRegionsPtr rleRegions(new RleRegions());
	ClassTypeGeneric typeConverter(classNamesFile);

	// Loop through all the new detections
		for (auto it = detections.begin(), end=detections.end(); it !=end; ++it){
			// Set the classID of the detected object.
			typeConverter.setId(it->classId);
			// Store the bounding boxes,class type,and its probability in regions
			regions->add(it->boundingBox,typeConverter.getClassString(),it->probability);
			// If masks are also available, store them aswell.
			if (this->hasMasks)
	        	rleRegions->add(it->rleRegion, typeConverter.getClassString(), it->probability);
			//std::cout<< it->boundingBox.x << " " << it->boundingBox.y << " " << it->boundingBox.height << " " << it->boundingBox.width << std::endl;
			LOG(INFO)<< typeConverter.getClassString() << ": " << it->probability << std::endl;
		}

		/*
			Store the image on which the detection is done,
			the set of detections(both the bounding boxes and masks)
			in the variable Sample.
		*/
	sample.setColorImage(image);
	sample.setRectRegions(regions);
  sample.setRleRegions(rleRegions);
	sample.SetMousy(this->mousy);
	this->mousy=false;
	return sample;
}

/*
This function converts the output from python scripts into a fromat compatible
DetectionSuite to read bounding boxes, classes and detection scores, which are
drawn on the image to show detections.
*/

void TensorFlowInferencer::output_result(int num_detections, int width, int height, PyObject* bounding_boxes, PyObject* detection_scores, PyObject* classIds, PyObject* pDetection_masks )
{
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
                throw std::invalid_argument("Returned Mask by tensorflow doesn't have 2 dimensions");
            }
            mask_shape = (long long int*) PyArray_SHAPE(detection_masks_cont);
        }

		PyArrayObject* bounding_boxes_cont = PyArray_GETCONTIGUOUS( (PyArrayObject*) bounding_boxes );

		PyArrayObject* detection_scores_cont = PyArray_GETCONTIGUOUS( (PyArrayObject*) detection_scores );

		PyArrayObject* classIds_cont = PyArray_GETCONTIGUOUS( (PyArrayObject*) classIds );


		float* bounding_box_data = (float*) bounding_boxes_cont->data; // not copying data
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
			detections[i].classId = classIds_data[classes++] - 1;  // In TensorFlow id's start from 1 whereas detectionsuite starts from 0s
			detections[i].probability = detection_scores_data[scores++];

			detections[i].boundingBox.y = bounding_box_data[boxes++] * height;

			detections[i].boundingBox.x = bounding_box_data[boxes++] * width;

			detections[i].boundingBox.height = bounding_box_data[boxes++] * height - detections[i].boundingBox.y;

			detections[i].boundingBox.width = bounding_box_data[boxes++] * width - detections[i].boundingBox.x;

            if (this->hasMasks) {

                cv::Mat image_mask(height, width, CV_8UC1, cv::Scalar(0));

                cv::Mat mask = cv::Mat(mask_shape[1], mask_shape[2], CV_32F, detection_masks_data + i*mask_shape[1]*mask_shape[2]);
				//std::cout << "Showing mask for 5 seconds" << mask_shape[0] << " " << mask_shape[1] << " " << mask_shape[2] << " " << mask_shape[3] << '\n';
                cv::Mat mask_r;

                cv::resize(mask, mask_r, cv::Size(detections[i].boundingBox.width, detections[i].boundingBox.height));
                cv::Mat mask_char;
                mask_r.convertTo(mask_char, CV_8U, 255);
                cv::threshold(mask_char, mask_char, 127, 255, cv::THRESH_BINARY);

                mask_char.copyTo(image_mask(cv::Rect(detections[i].boundingBox.x,detections[i].boundingBox.y,detections[i].boundingBox.width, detections[i].boundingBox.height)));
								RLE forMask;
								cv::Mat t_mask = image_mask.t();
                rleEncode( &forMask, t_mask.data, t_mask.cols, t_mask.rows, 1 );
                //std::cout << rleToString(&forMask) << '\n';
                //cv::Mat matrix_decoded(image_mask.rows, image_mask.cols, CV_8U);

                //matrix_decoded = matrix_decoded * 255;
                //rleDecode(&forMask, matrix_decoded.data , 1);
                //matrix_decoded = matrix_decoded * 255;
                //cv::findContours( mask_char.clone(), contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cv::Point(detections[i].boundingBox.x, detections[i].boundingBox.y) );
                //std::cout << "Outer Vector Size:" << contours.size() << '\n';
                //std::cout << "Inner Vector Size:" << contours[0].size() << '\n';
                detections[i].rleRegion = forMask;
                //cv::drawContours(image_mask, contours, -1, cv::Scalar(255, 0, 0), 2, 8);

                //cv::imshow("mask", mask_char);
                //cv::imshow("decoded mask", matrix_decoded);
                //cv::waitKey(5000);
            }


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

int TensorFlowInferencer::gettfInferences(const cv::Mat& image, double confidence_threshold) {


	int i, num_detections, dims, sizes[3];

	// Check if it is RGB image and store the dimensions of the image in "sizes"
	if (image.channels() == 3) {
		dims = 3;
		sizes[0] = image.rows;
		sizes[1] = image.cols;
		sizes[2] = image.channels();

	// Check if it is a monochromatic image and repeat the above.
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
		num_detections = _PyInt_AsInt( PyDict_GetItem(pValue, PyString_FromString("num_detections") ) );
		printf("Num Detections: %d\n",  num_detections );
		PyObject* pBounding_boxes = PyDict_GetItem(pValue, PyString_FromString("detection_boxes") );
		PyObject* pDetection_scores = PyDict_GetItem(pValue, PyString_FromString("detection_scores") );
		PyObject* classIds = PyDict_GetItem(pValue, PyString_FromString("detection_classes") );
        PyObject* key = PyString_FromString("detection_masks");
        if (PyDict_Contains(pValue, key)) {
            PyObject* pDetection_masks = PyDict_GetItem(pValue, PyString_FromString("detection_masks") );
            output_result(num_detections, image.cols, image.rows, pBounding_boxes, pDetection_scores, classIds, pDetection_masks);
        } else {
            output_result(num_detections, image.cols, image.rows, pBounding_boxes, pDetection_scores, classIds);
        }
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
