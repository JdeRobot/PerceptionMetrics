//
// Created by frivas on 21/01/17.
//


#include <iostream>
#include <string>
#include <Utils/SampleGenerationApp.h>
#include <DatasetConverters/readers/GenericDatasetReader.h>
#include <glog/logging.h>

class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.push_back("inputPath");
        this->requiredArguments.push_back("readerImplementation");
        this->requiredArguments.push_back("readerNames");


    };
    void operator()(){
        YAML::Node  inputPathNode=this->config.getNode("inputPath");
        YAML::Node  readerImplementationNode = this->config.getNode("readerImplementation");
        YAML::Node  readerNamesNode = this->config.getNode("readerNames");


        GenericDatasetReaderPtr reader;
        if (inputPathNode.IsSequence()) {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPathNode.as<std::vector<std::string>>(),readerNamesNode.as<std::string>(), readerImplementationNode.as<std::string>(), true));
        }
        else {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPathNode.as<std::string>(),readerNamesNode.as<std::string>(), readerImplementationNode.as<std::string>(), true));
        }



        Sample sample;
        while (reader->getReader()->getNextSample(sample)){
            LOG(INFO) << "number of elements: " << sample.getRectRegions()->getRegions().size() << std::endl;
            cv::Mat image =sample.getSampledColorImage();
            cv::imshow("Viewer", image);
            cv::waitKey(0);
        }

    };
};

int main (int argc, char* argv[])
{

    MyApp myApp(argc,argv);
    myApp.process();
}
