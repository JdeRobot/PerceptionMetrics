

#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/highgui/highgui.hpp>
#include "DatasetConverters/liveReaders/RecorderReader.h"
#include "GenerationUtils/DepthForegroundSegmentator.h"
#include "GenerationUtils/DetectionsValidator.h"
#include <glog/logging.h>
#include <Utils/SampleGenerationApp.h>
#include <FrameworkEvaluator/FrameworkInferencer.h>

#ifdef DARKNET_ACTIVE
#include <FrameworkEvaluator/DarknetInferencer.h>
#endif

#include <DatasetConverters/readers/GenericDatasetReader.h>


class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.emplace_back("outputPath");
        this->requiredArguments.emplace_back("reader");
        this->requiredArguments.emplace_back("detector");


    };
    virtual void operator()(){
        Key outputPath=this->config->getKey("outputPath");
        Key reader=this->config->getKey("reader");
        Key detectorKey = this->config->getKey("detector");
        Key colorImagesPathKey;
        Key depthImagesPathKey;
        Key dataPath;



        if  (reader.getValue() == "recorder"){
            colorImagesPathKey = this->config->getKey("colorImagesPath");
            depthImagesPathKey = this->config->getKey("depthImagesPath");
        }
        else{
            dataPath = this->config->getKey("dataPath");
        }




        //todo include in upper class
        std::vector<std::string> detectorOptions;
        detectorOptions.push_back("pentalo-bg");
        detectorOptions.push_back("deepLearning");
        detectorOptions.push_back("datasetReader");



        if (std::find(detectorOptions.begin(),detectorOptions.end(),detectorKey.getValue())== detectorOptions.end()){
            LOG(ERROR) << detectorKey.getValue() << " is nor supported";
            exit(1);
        }


        if (detectorKey.getValue()=="pentalo-bg") {

            RecorderReader converter(colorImagesPathKey.getValue(), depthImagesPathKey.getValue());
            DepthForegroundSegmentator segmentator;


            DetectionsValidator validator(outputPath.getValue());
            cv::Mat previousImage;
            int counter = 0;
            int maxElements = converter.getNumSamples();
            Sample sample;
            while (converter.getNextSample(sample)) {
                counter++;
                std::stringstream ss;
                ss << counter << "/" << maxElements;
                LOG(INFO) << "Processing [" + ss.str() + "]";
                cv::Mat colorImage = sample.getColorImage().clone();
                cv::cvtColor(colorImage, colorImage, CV_RGB2BGR);
                if (!previousImage.empty()) {
                    cv::Mat diff;
                    cv::absdiff(colorImage, previousImage, diff);
                    auto val = cv::sum(cv::sum(diff));
                    if (val[0] < 1000) {
                        continue;
                    }
                }
                colorImage.copyTo(previousImage);
                cv::Mat depthImage = sample.getDepthImage().clone();
                std::vector<std::vector<cv::Point>> detections = segmentator.process(depthImage);

                validator.validate(colorImage, depthImage, detections);
            }
        }
        else if (detectorKey.getValue()=="deepLearning") {
            Key inferencerImplementationKey=this->config->getKey("inferencerImplementation");
            Key inferencerNamesKey=this->config->getKey("inferencerNames");
            Key inferencerConfigKey=this->config->getKey("inferencerConfig");
            Key inferencerWeightsKey=this->config->getKey("inferencerWeights");


            RecorderReaderPtr converter;
            if (reader.getValue() == "recorder-rgbd") {
                converter=RecorderReaderPtr( new RecorderReader(dataPath.getValue()));
            }
            else{
                converter=RecorderReaderPtr( new RecorderReader(colorImagesPathKey.getValue(), depthImagesPathKey.getValue()));
            }

            FrameworkInferencerPtr inferencer;

            if (inferencerImplementationKey.getValue()=="yolo") {
#ifdef DARKNET_ACTIVE
                inferencer = DarknetInferencerPtr( new DarknetInferencer(inferencerConfigKey.getValue(), inferencerWeightsKey.getValue(), inferencerNamesKey.getValue()));
#else
                LOG(WARNING) << "Darknet inferencer is not available";
#endif
            }
            else{
                LOG(WARNING) << inferencerImplementationKey.getValue() + " is not a valid inferencer implementation";
            }

            DetectionsValidator validator(outputPath.getValue());
            int maxElements = converter->getNumSamples();
            Sample sample;
            int counter=0;
            int skipSamples=10;
            std::random_device rd; // obtain a random number from hardware
            std::mt19937 eng(rd()); // seed the generator
            std::uniform_int_distribution<> distr(5, skipSamples);

            if (maxElements==0){
                LOG(ERROR) << "Empty sample data";
                exit(1);
            }

            while (converter->getNextSample(sample)) {
                int samples_to_skip=distr(eng);
                std::cout << "Skipping. " << samples_to_skip << std::endl;
                bool validSample=false;
                for (size_t i = 0; i < samples_to_skip; i++){
                    validSample=converter->getNextSample(sample);
                }
                if (!validSample)
                    break;


                counter++;
                std::stringstream ss;
                ss << counter << "/" << maxElements;
                LOG(INFO) << "Processing [" + ss.str() + "]";

                Sample detectedSample = inferencer->detect(sample.getColorImage());
                detectedSample.setColorImage(sample.getColorImage());
                detectedSample.setDepthImage(sample.getDepthImage());


                validator.validate(detectedSample);


            }
        }
        else if(detectorKey.getValue()=="datasetReader"){
            Key readerNamesKey=this->config->getKey("readerNames");
            //readerImplementationGT
            GenericDatasetReaderPtr readerImp(new GenericDatasetReader(dataPath.getValue(),readerNamesKey.getValue(), reader.getValue()));


            DetectionsValidator validator(outputPath.getValue(),1.5);

            int maxElements = -1;
            Sample sample;
            int counter=0;
            int skipSamples=10;
            std::random_device rd; // obtain a random number from hardware
            std::mt19937 eng(rd()); // seed the generator
            std::uniform_int_distribution<> distr(5, skipSamples);

            while (readerImp->getReader()->getNextSample(sample)) {
                int samples_to_skip=distr(eng);
                std::cout << "Skipping. " << samples_to_skip << std::endl;
                bool validSample=false;
                for (size_t i = 0; i < samples_to_skip; i++){
                    validSample=readerImp->getReader()->getNextSample(sample);
                }
                if (!validSample)
                    break;


                counter++;
                std::stringstream ss;
                ss << counter << "/" << maxElements;
                LOG(INFO) << "Processing [" + ss.str() + "]";




                validator.validate(sample);


            }

        }
    };
};



int main (int argc, char* argv[])
{

    MyApp myApp(argc,argv);
    myApp.process();
}
