#include <Common/Sample.h>
#include <DatasetConverters/ClassTypeGeneric.h>
#include "CaffeInferencer.h"

CaffeInferencer::CaffeInferencer(const std::string &netConfig, const std::string &netWeights,const std::string& classNamesFile): netConfig(netConfig),netWeights(netWeights) {
    this->classNamesFile=classNamesFile;
    this->netConfig=netConfig;
    this->netWeights=netWeights;

    this->confThreshold = 0.4;
    //float scale = parser.get<float>("scale");
    //Scalar mean = parser.get<Scalar>("mean");
    //bool swapRB = parser.get<bool>("rgb");
    //int inpWidth = parser.get<int>("width");
    //int inpHeight = parser.get<int>("height");

    // Open file with classes names.
    /*if (parser.has("classes"))
    {
        std::string file = parser.get<String>("classes");
        std::ifstream ifs(file.c_str());
        if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + file + " not found");
        std::string line;
        while (std::getline(ifs, line))
        {
            classes.push_back(line);
        }
    }*/

    // Load a model.
    //CV_Assert(parser.has("model"));
    this->net = cv::dnn::readNetFromCaffe(this->netConfig, this->netWeights);
    //net.setPreferableBackend(parser.get<int>("backend"));
    //net.setPreferableTarget(parser.get<int>("target"));



}

Sample CaffeInferencer::detectImp(const cv::Mat &image) {

    cv::Mat blob;
    cv::Mat rgbImage;

    cv::cvtColor(image,rgbImage,CV_BGR2RGB);


    cv::Size inpSize(rgbImage.cols, rgbImage.rows);
    blob = cv::dnn::blobFromImage(rgbImage,1.0, inpSize,cv::Scalar(), true, false);
    //blobFromImage(frame, blob, scale, inpSize, mean, swapRB, false);
    // Run a model.
    this->net.setInput(blob);
    if (this->net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
    {
        cv::resize(rgbImage, rgbImage, inpSize);
        cv::Mat imInfo = (cv::Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
        this->net.setInput(imInfo, "im_info");
    }
    std::vector<cv::Mat> outs;
    this->net.forward(outs, getOutputsNames());

    postprocess(outs);

    // Put efficiency information.
    /*std::vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    std::string label = format("Inference time: %.2f ms", t);
    putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
    */


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

void CaffeInferencer::postprocess(const std::vector<cv::Mat>& outs)
{
    static std::vector<int> outLayers = this->net.getUnconnectedOutLayers();
    static std::string outLayerType = this->net.getLayer(outLayers[0])->type;

    if (this->net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        cv::CV_Assert(outs.size() == 1);
        float* data = (float*)outs[0].data;
        for (size_t i = 0; i < outs[0].total(); i += 7)
        {
            float confidence = data[i + 2];
            if (confidence > confThreshold)
            {
                detections.push_back(detection());
                detections[i].classId = (int)(data[i + 1]) - 1;
                detections[i].probability = confidence;
                detections[i].boundingBox.x = (int)data[i + 3];
                detections[i].boundingBox.y = (int)data[i + 4];

                detections[i].boundingBox.width = (int)data[i + 5] - detections[i].boundingBox.x;

                detections[i].boundingBox.height = (int)data[i + 6] - detections[i].boundingBox.y;

            }
        }

    }
    else if (outLayerType == "DetectionOutput")
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        cv::CV_Assert(outs.size() == 1);
        float* data = (float*)outs[0].data;
        for (size_t i = 0; i < outs[0].total(); i += 7)
        {
            float confidence = data[i + 2];
            if (confidence > confThreshold)
            {
                detections.push_back(detection());
                detections[i].classId = (int)(data[i + 1]) - 1;
                detections[i].probability = confidence;
                detections[i].boundingBox.x = (int)data[i + 3];
                detections[i].boundingBox.y = (int)data[i + 4];

                detections[i].boundingBox.width = (int)data[i + 5] - detections[i].boundingBox.x;

                detections[i].boundingBox.height = (int)data[i + 6] - detections[i].boundingBox.y;

            }
        }
    }
    else if (outLayerType == "Region")
    {
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                cv::Point classIdPoint;
                double confidence;
                cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }
        std::vector<int> indices;
        cv::NMSBoxes(boxes, confidences, confThreshold, 0.4f, indices);
        for (size_t i = 0; i < indices.size(); ++i)
        {
            int idx = indices[i];
            cv::Rect box = boxes[idx];

            detections.push_back(detection());
            detections[i].classId = classIds[idx];
            detections[i].probability = confidences[idx];
            detections[i].boundingBox.x = box.x;
            detections[i].boundingBox.y = box.y;

            detections[i].boundingBox.width = box.width;

            detections[i].boundingBox.height = box.height;

            //drawPred(classIds[idx], confidences[idx], box.x, box.y,
            //         box.x + box.width, box.y + box.height, frame);
        }
    }
    else
        cv::CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);
}

/*void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

    std::string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}
*/
std::vector<cv::String> CaffeInferencer::getOutputsNames()
{
    static std::vector<String> names;
    if (names.empty())
    {
        std::vector<int> outLayers = this->net.getUnconnectedOutLayers();
        std::vector<cv::String> layersNames = this->net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}
