//
// Created by frivas on 21/01/17.
//


#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <glog/logging.h>
#include <DatasetConverters/readers/OwnDatasetReader.h>
#include <DatasetConverters/writers/YoloDatasetWriter.h>
#include <DatasetConverters/readers/YoloDatasetReader.h>


#include <iostream>
#include <string>
#include <Utils/SampleGenerationApp.h>
#include <DatasetConverters/readers/GenericDatasetReader.h>
#include <DatasetConverters/writers/GenericDatasetWriter.h>


class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.push_back("inputPath");
        this->requiredArguments.push_back("outputPath");
        this->requiredArguments.push_back("readerImplementation");
        this->requiredArguments.push_back("writerImplementation");
        this->requiredArguments.push_back("readerNames");
        this->requiredArguments.push_back("writeImages");


    };
    void operator()(){
        YAML::Node inputPathNode=this->config.getNode("inputPath");
        YAML::Node readerImplementationNode = this->config.getNode("readerImplementation");
        YAML::Node writerImplementationNode = this->config.getNode("writerImplementation");
        YAML::Node outputPathNode = this->config.getNode("outputPath");
        YAML::Node readerNamesNode = this->config.getNode("readerNames");
        YAML::Node writeImages = this->config.getNode("writeImages");

        GenericDatasetReaderPtr reader;
        if (inputPathNode.IsSequence()) {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPathNode.as<std::vector<std::string>>(), readerNamesNode.as<std::string>(), readerImplementationNode.as<std::string>(), writeImages.as<bool>()));
        }
        else {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPathNode.as<std::string>(),readerNamesNode.as<std::string>(), readerImplementationNode.as<std::string>(), writeImages.as<bool>()));
        }

        auto readerPtr = reader->getReader();

//        std::vector<std::string> idsToFilter;
//        idsToFilter.push_back("person");
//        idsToFilter.push_back("person-falling");
//        idsToFilter.push_back("person-fall");
//        readerPtr->filterSamplesByID(idsToFilter);
//        readerPtr->printDatasetStats();


        GenericDatasetWriterPtr writer( new GenericDatasetWriter(outputPathNode.as<std::string>(),readerPtr,writerImplementationNode.as<std::string>()));
        writer->getWriter()->process(writeImages.as<bool>());
    };
};

int main (int argc, char* argv[])
{

    MyApp myApp(argc,argv);
    myApp.process();
}

/*void extractPersonsFromYolo(const std::string& dataSetPath){
    YoloDatasetReader reader(dataSetPath);

    std::vector<std::string> idsToFilter;
    idsToFilter.push_back("person");


    std::cout << "Samples before: " << reader.getNumberOfElements() << std::endl;
    reader.filterSamplesByID(idsToFilter);
    std::cout << "Samples after: " << reader.getNumberOfElements() << std::endl;
    YoloDatasetWriter converter("converter_output", reader);
    converter.process(true);
}



int main (int argc, char* argv[]) {

    ViewerAguments args;
    parse_arguments(argc,argv,args);


    Logger::getInstance()->setLevel(Logger::INFO);
    Logger::getInstance()->info("Reviewing " + args.path);


    extractPersonsFromYolo(args.path);

    OwnDatasetReader reader(args.path);

    YoloDatasetWriter converter("converter_output", reader);
    converter.process(true);
}
*/
