#include <Common/Sample.h>
#include <DatasetConverters/ClassTypeGeneric.h>
#include "CaffeInferencer.h"
#include <glog/logging.h>

CaffeInferencer::CaffeInferencer(const std::string &netConfig, const std::string &netWeights,const std::string& classNamesFile, std::map<std::string, std::string>* inferencerParamsMap): netConfig(netConfig),netWeights(netWeights) {
    this->classNamesFile=classNamesFile;
    this->netConfig=netConfig;
    this->netWeights=netWeights;

    //this->confThreshold = std::stof(inferencerParamsMap->at("conf_thresh"));
    this->scaling_factor = std::stof(inferencerParamsMap->at("scaling_factor"));
    this->mean_sub = cv::Scalar(std::stof(inferencerParamsMap->at("mean_sub_blue")), std::stof(inferencerParamsMap->at("mean_sub_green")), std::stof(inferencerParamsMap->at("mean_sub_red")));
    this->swapRB = inferencerParamsMap->at("useRGB") == "true";
    this->inpWidth = std::stof(inferencerParamsMap->at("inpWidth"));
    this->inpHeight = std::stof(inferencerParamsMap->at("inpHeight"));

    // Load a model.
    //CV_Assert(parser.has("model"));
    this->net = cv::dnn::readNetFromCaffe(this->netConfig, this->netWeights);
    //net.setPreferableBackend(parser.get<int>("backend"));
    //net.setPreferableTarget(parser.get<int>("target"));
    net.setPreferableBackend(0);
    net.setPreferableTarget(0);



}

Sample CaffeInferencer::detectImp(const cv::Mat &image, double confidence_threshold) {

    cv::Mat blob;

    cv::Mat rgbImage = image;
    this->detections.clear();


    cv::Size inpSize(this->inpWidth, this->inpHeight);
    blob = cv::dnn::blobFromImage(rgbImage, this->scaling_factor, inpSize, this->mean_sub, false, false);
    //blobFromImage(frame, blob, scale, inpSize, mean, swapRB, false);
    // Run a model.


    this->net.setInput(blob);


    if (this->net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
    {
        // For faster RCNN, same size images are being passed
        //cv::resize(rgbImage, rgbImage, inpSize);
        cv::Mat imInfo = (cv::Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
        this->net.setInput(imInfo, "im_info");
    }
    std::vector<cv::Mat> outs;


    this->net.forward(outs, getOutputsNames());


    postprocess(outs, rgbImage, confidence_threshold);


    Sample sample;
    RectRegionsPtr regions(new RectRegions());
    ClassTypeGeneric typeConverter(classNamesFile);

    for (auto it = detections.begin(), end=detections.end(); it !=end; ++it){

		typeConverter.setId(it->classId);
		regions->add(it->boundingBox,typeConverter.getClassString(), it->probability);
		LOG(INFO)<< it->boundingBox.x << " " << it->boundingBox.y << " " << it->boundingBox.height << " " << it->boundingBox.width << std::endl;
		LOG(INFO)<< typeConverter.getClassString() << ": " << it->probability << std::endl;
	}
    sample.setColorImage(image);
    sample.setRectRegions(regions);
    return sample;
}

void CaffeInferencer::postprocess(const std::vector<cv::Mat>& outs, cv::Mat & image, double confidence_threshold)
{
    static std::vector<int> outLayers = this->net.getUnconnectedOutLayers();
    static std::string outLayerType = this->net.getLayer(outLayers[0])->type;

    if (this->net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        //cv::CV_Assert(outs.size() == 1);
        assert(outs.size() == 1);
        float* data = (float*)outs[0].data;
        int count = 0;
        for (size_t i = 0; i < outs[0].total(); i += 7)
        {
            float confidence = data[i + 2];
            if (confidence >= confidence_threshold)
            {
                detections.push_back(detection());
                detections[count].classId = (int)(data[i + 1]) - 1;
                detections[count].probability = confidence;
                detections[count].boundingBox.x = (((int)data[i + 3] * image.cols) / inpWidth);
                detections[count].boundingBox.y = (((int)data[i + 4] * image.rows) / inpHeight);

                detections[count].boundingBox.width = (((int)data[i + 5] * image.cols) / inpWidth) - detections[count].boundingBox.x;

                detections[count].boundingBox.height = (((int)data[i + 6] * image.rows) / inpHeight) - detections[count].boundingBox.y;

                count++;
            }

        }

    }
    else if (outLayerType == "DetectionOutput")
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        assert(outs.size() == 1);
        float* data = (float*)outs[0].data;
        int count = 0;
        for (size_t i = 0; i < outs[0].total(); i += 7)
        {
            float confidence = data[i + 2];
            if (confidence >= confidence_threshold)
            {
                detections.push_back(detection());
                detections[count].classId = (int)(data[i + 1]) - 1;
                detections[count].probability = confidence;
                LOG(INFO) << data[i + 3] << '\n';
                detections[count].boundingBox.x = (int)(data[i + 3] * image.cols);
                detections[count].boundingBox.y = (int)(data[i + 4] * image.rows);

                detections[count].boundingBox.width = (int)(data[i + 5] * image.cols )- detections[count].boundingBox.x;

                detections[count].boundingBox.height = (int)(data[i + 6] * image.rows ) - detections[count].boundingBox.y;

                count++;

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
                if (confidence >= confidence_threshold)
                {
                    int centerX = (int)(data[0] * image.cols);
                    int centerY = (int)(data[1] * image.rows);
                    int width = (int)(data[2] * image.cols);
                    int height = (int)(data[3] * image.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, 0.4f, indices);
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
        throw std::invalid_argument("Unknown output layer type: " + outLayerType);
}

std::vector<cv::String> CaffeInferencer::getOutputsNames()
{

    if (this->names.empty())
    {
        std::vector<int> outLayers = this->net.getUnconnectedOutLayers();
        std::vector<cv::String> layersNames = this->net.getLayerNames();
        this->names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            this->names[i] = layersNames[outLayers[i] - 1];
    }
    return this->names;
}
