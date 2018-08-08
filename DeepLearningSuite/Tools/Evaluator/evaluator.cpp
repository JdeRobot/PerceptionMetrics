//
// Created by frivas on 1/02/17.
//


#include <iostream>
#include <string>
#include <Utils/SampleGenerationApp.h>
#include <DatasetConverters/readers/GenericDatasetReader.h>
#include <FrameworkEvaluator/DetectionsEvaluator.h>
#include <FrameworkEvaluator/StatsWriter.h>


class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.push_back("outputPath");
        this->requiredArguments.push_back("inputPathGT");
        this->requiredArguments.push_back("inputPathDetection");
        this->requiredArguments.push_back("readerImplementationGT");
        this->requiredArguments.push_back("readerImplementationDetection");
        this->requiredArguments.push_back("readerNames");
        this->requiredArguments.push_back("iouType");


    };
    void operator()(){
        YAML::Node outputPath=this->config.getNode("outputPath");
        YAML::Node inputPathGT=this->config.getNode("inputPathGT");
        YAML::Node inputPathDetection=this->config.getNode("inputPathDetection");
        YAML::Node readerImplementationGTKey=this->config.getNode("readerImplementationGT");
        YAML::Node readerImplementationDetectionKey=this->config.getNode("readerImplementationDetection");
        YAML::Node readerNamesKey=this->config.getNode("readerNames");
        std::string iouType = this->config.asString("iouType");


        GenericDatasetReaderPtr readerGT(new GenericDatasetReader(inputPathGT.as<std::string>(),readerNamesKey.as<std::string>(), readerImplementationGTKey.as<std::string>(), false));
        GenericDatasetReaderPtr readerDetection(new GenericDatasetReader(inputPathDetection.as<std::string>(),readerNamesKey.as<std::string>(), readerImplementationDetectionKey.as<std::string>(), false));


        DetectionsEvaluatorPtr evaluator(new DetectionsEvaluator(readerGT->getReader(),readerDetection->getReader(),true));
        //todo ñapa

        bool isIouTypeBbox;

        if (iouType == "segm" || iouType == "bbox") {
            isIouTypeBbox = iouType == "bbox";
        } else {
            throw std::invalid_argument("Evaluation iouType can either be 'segm' or 'bbox'\n");
        }

        evaluator->evaluate(isIouTypeBbox);
        evaluator->accumulateResults();


        std::string mywriterFile("Evaluation Results.csv" );

        StatsWriter writer(readerGT->getReader(), mywriterFile);

        writer.writeInferencerResults("Detection Dataset", evaluator);

        writer.saveFile();


    };
};





int main (int argc, char* argv[]) {
    MyApp myApp(argc,argv);
    myApp.process();
}
