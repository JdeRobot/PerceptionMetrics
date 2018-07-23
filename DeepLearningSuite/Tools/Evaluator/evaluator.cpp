//
// Created by frivas on 1/02/17.
//


#include <iostream>
#include <string>
#include <Utils/SampleGenerationApp.h>
#include <DatasetConverters/readers/GenericDatasetReader.h>
#include <FrameworkEvaluator/DetectionsEvaluator.h>


class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.push_back("outputPath");
        this->requiredArguments.push_back("inputPathGT");
        this->requiredArguments.push_back("inputPathDetection");
        this->requiredArguments.push_back("readerImplementationGT");
        this->requiredArguments.push_back("readerImplementationDetection");
        this->requiredArguments.push_back("readerNames");


    };
    void operator()(){
        YAML::Node outputPath=this->config.getNode("outputPath");
        YAML::Node inputPathGT=this->config.getNode("inputPathGT");
        YAML::Node inputPathDetection=this->config.getNode("inputPathDetection");
        YAML::Node readerImplementationGTKey=this->config.getNode("readerImplementationGT");
        YAML::Node readerImplementationDetectionKey=this->config.getNode("readerImplementationDetection");
        YAML::Node readerNamesKey=this->config.getNode("readerNames");



        GenericDatasetReaderPtr readerGT(new GenericDatasetReader(inputPathGT.as<std::string>(),readerNamesKey.as<std::string>(), readerImplementationGTKey.as<std::string>(), false));
        GenericDatasetReaderPtr readerDetection(new GenericDatasetReader(inputPathDetection.as<std::string>(),readerNamesKey.as<std::string>(), readerImplementationDetectionKey.as<std::string>(), false));


        DetectionsEvaluatorPtr evaluator(new DetectionsEvaluator(readerGT->getReader(),readerDetection->getReader(),true));
        //todo ñapa
        evaluator->addValidMixClass("person", "person-falling");
        evaluator->addValidMixClass("person", "person-fall");
        evaluator->addClassToDisplay("person");
        evaluator->addClassToDisplay("person-falling");
        evaluator->addClassToDisplay("person-fall");
        evaluator->evaluate();
        evaluator->accumulateResults();


    };
};





int main (int argc, char* argv[]) {
    MyApp myApp(argc,argv);
    myApp.process();
}
