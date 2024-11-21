//
// Created by frivas on 1/02/17.
//


#include <iostream>
#include <Utils/SampleGenerationApp.h>
#include <DatasetConverters/readers/OwnDatasetReader.h>
#include <FrameworkEvaluator/MassInferencer.h>
#include <DatasetConverters/readers/GenericDatasetReader.h>
#include <FrameworkEvaluator/GenericInferencer.h>


class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.push_back("inputPath");
        this->requiredArguments.push_back("outputPath");
        this->requiredArguments.push_back("readerImplementation");
        this->requiredArguments.push_back("inferencerImplementation");
        this->requiredArguments.push_back("inferencerConfig");
        this->requiredArguments.push_back("inferencerWeights");
        this->requiredArguments.push_back("inferencerNames");
        this->requiredArguments.push_back("readerNames");
    };
    void operator()(){
        YAML::Node inputPath=this->config.getNode("inputPath");
        YAML::Node outputPath=this->config.getNode("outputPath");

        YAML::Node readerImplementationKey = this->config.getNode("readerImplementation");
        YAML::Node infererImplementationKey = this->config.getNode("inferencerImplementation");
        YAML::Node inferencerConfigKey = this->config.getNode("inferencerConfig");
        YAML::Node inferencerWeightsKey = this->config.getNode("inferencerWeights");
        YAML::Node inferencerNamesKey = this->config.getNode("inferencerNames");
        YAML::Node readerNamesKey = this->config.getNode("readerNames");

        GenericDatasetReaderPtr reader;
        if (inputPath.IsSequence()) {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPath.as<std::vector<std::string>>(),readerNamesKey.as<std::string>(), readerImplementationKey.as<std::string>(), true));
        }
        else {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPath.as<std::string>(),readerNamesKey.as<std::string>(), readerImplementationKey.as<std::string>(), true));
        }


        GenericInferencerPtr inferencer(new GenericInferencer(inferencerConfigKey.as<std::string>(),inferencerWeightsKey.as<std::string>(),inferencerNamesKey.as<std::string>(),infererImplementationKey.as<std::string>()));
        MassInferencer massInferencer(reader->getReader(),inferencer->getInferencer(),outputPath.as<std::string>(), true);
        massInferencer.process(false);

    };
};



int main (int argc, char* argv[]) {

    MyApp myApp(argc,argv);
    myApp.process();
}
