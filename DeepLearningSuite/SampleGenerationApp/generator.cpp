

#include <iostream>
#include <random>
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
        YAML::Node outputPath=this->config.getNode("outputPath");
        YAML::Node reader=this->config.getNode("reader");
        YAML::Node detectorKey = this->config.getNode("detector");
        YAML::Node colorImagesPathKey;
        YAML::Node depthImagesPathKey;
        YAML::Node dataPath;



        if  (reader.as<std::string>() == "recorder"){
            colorImagesPathKey = this->config.getNode("colorImagesPath");
            depthImagesPathKey = this->config.getNode("depthImagesPath");
        }
        else{
            dataPath = this->config.getNode("dataPath");
        }




        //todo include in upper class
        std::vector<std::string> detectorOptions;
        detectorOptions.push_back("pentalo-bg");
        detectorOptions.push_back("deepLearning");
        detectorOptions.push_back("datasetReader");



        if (std::find(detectorOptions.begin(),detectorOptions.end(),detectorKey.as<std::string>())== detectorOptions.end()){
            LOG(ERROR) << detectorKey.as<std::string>() << " is nor supported";
            exit(1);
        }


        if (detectorKey.as<std::string>()=="pentalo-bg") {

            RecorderReader converter(colorImagesPathKey.as<std::string>(), depthImagesPathKey.as<std::string>());
            DepthForegroundSegmentator segmentator;


            DetectionsValidator validator(outputPath.as<std::string>());
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
                cv::cvtColor(colorImage, colorImage, cv::COLOR_RGB2BGR);
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
        else if (detectorKey.as<std::string>()=="deepLearning") {
            YAML::Node inferencerImplementationKey=this->config.getNode("inferencerImplementation");
            YAML::Node inferencerNamesKey=this->config.getNode("inferencerNames");
            YAML::Node inferencerConfigKey=this->config.getNode("inferencerConfig");
            YAML::Node inferencerWeightsKey=this->config.getNode("inferencerWeights");


            RecorderReaderPtr converter;
            if (reader.as<std::string>() == "recorder-rgbd") {
                converter=RecorderReaderPtr( new RecorderReader(dataPath.as<std::string>()));
            }
            else{
                converter=RecorderReaderPtr( new RecorderReader(colorImagesPathKey.as<std::string>(), depthImagesPathKey.as<std::string>()));
            }

            FrameworkInferencerPtr inferencer;

            if (inferencerImplementationKey.as<std::string>()=="yolo") {
#ifdef DARKNET_ACTIVE
                inferencer = DarknetInferencerPtr( new DarknetInferencer(inferencerConfigKey.as<std::string>(), inferencerWeightsKey.as<std::string>(), inferencerNamesKey.as<std::string>()));
#else
                LOG(WARNING) << "Darknet inferencer is not available";
#endif
            }
            else{
                LOG(WARNING) << inferencerImplementationKey.as<std::string>() + " is not a valid inferencer implementation";
            }

            DetectionsValidator validator(outputPath.as<std::string>());
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
                LOG(WARNING) << "Skipping. " << samples_to_skip << std::endl;
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

                double thresh = 0.2;
                Sample detectedSample = inferencer->detect(sample.getColorImage(), thresh);
                detectedSample.setColorImage(sample.getColorImage());
                detectedSample.setDepthImage(sample.getDepthImage());


                validator.validate(detectedSample);


            }
        }
        else if(detectorKey.as<std::string>()=="datasetReader"){
            YAML::Node readerNamesKey=this->config.getNode("readerNames");
            //readerImplementationGT
            GenericDatasetReaderPtr readerImp(new GenericDatasetReader(dataPath.as<std::string>(),readerNamesKey.as<std::string>(), reader.as<std::string>(), true));


            DetectionsValidator validator(outputPath.as<std::string>(),1.5);

            int maxElements = -1;
            Sample sample;
            int counter=0;
            int skipSamples=10;
            std::random_device rd; // obtain a random number from hardware
            std::mt19937 eng(rd()); // seed the generator
            std::uniform_int_distribution<> distr(5, skipSamples);

            while (readerImp->getReader()->getNextSample(sample)) {
                int samples_to_skip=distr(eng);
                LOG(WARNING) << "Skipping. " << samples_to_skip << std::endl;
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
