//
// Created by frivas on 5/02/17.
//

#include <iostream>
#include <string>
#include <Utils/SampleGenerationApp.h>
#include <DatasetConverters/readers/GenericDatasetReader.h>
#include <FrameworkEvaluator/DetectionsEvaluator.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <glog/logging.h>
#include <DatasetConverters/writers/GenericDatasetWriter.h>


class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.push_back("inputPath");
        this->requiredArguments.push_back("readerImplementation");
        this->requiredArguments.push_back("outputPath");
        this->requiredArguments.push_back("trainRatio");
        this->requiredArguments.push_back("readerNames");



    };
    void operator()(){
        YAML::Node inputPathNode=this->config.getNode("inputPath");
        YAML::Node outputPathNode=this->config.getNode("outputPath");
        YAML::Node readerImplementationNode = this->config.getNode("readerImplementation");
        YAML::Node writerImplementationNode = this->config.getNode("writerImplementation");
        YAML::Node trainRatioNode = this->config.getNode("trainRatio");
        YAML::Node readerNamesNode = this->config.getNode("readerNames");



        std::string trainPath=outputPathNode.as<std::string>() + "/train";
        std::string testPath=outputPathNode.as<std::string>() + "/test";


        GenericDatasetReaderPtr reader;
        if (inputPathNode.IsSequence()) {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPathNode.as<std::vector<std::string>>(),readerNamesNode.as<std::string>(), readerImplementationNode.as<std::string>(), true));
        }
        else {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPathNode.as<std::string>(),readerNamesNode.as<std::string>(),
                                             readerImplementationNode.as<std::string>(), true));
        }


        auto readerPtr = reader->getReader();

        std::vector<std::string> idsToFilter;
        idsToFilter.push_back("person");
        idsToFilter.push_back("person-falling");
        idsToFilter.push_back("person-fall");
        readerPtr->filterSamplesByID(idsToFilter);
        readerPtr->printDatasetStats();




        DatasetReaderPtr readerTest(new DatasetReader(true));
        DatasetReaderPtr readerTrain(new DatasetReader(true));


        int ratio=trainRatioNode.as<int>();

        Sample sample;
        int counter=0;
        while (readerPtr->getNextSample(sample)){
            if (counter <ratio){
                readerTrain->addSample(sample);
            }
            else{
                readerTest->addSample(sample);
            }
            counter++;
            counter= counter % 10;
        }

        LOG(INFO) << "Train: " << '\n';
        readerTrain->printDatasetStats();
        LOG(INFO) << "Test: " << '\n';
        readerTest->printDatasetStats();


        GenericDatasetWriterPtr writerTest( new GenericDatasetWriter(testPath,readerTest,writerImplementationNode.as<std::string>()));
        writerTest->getWriter()->process();

        GenericDatasetWriterPtr writerTrain( new GenericDatasetWriter(trainPath,readerTrain,writerImplementationNode.as<std::string>()));
        writerTrain->getWriter()->process();

    };
};

int main (int argc, char* argv[]) {

    MyApp myApp(argc, argv);
    myApp.process();
}
