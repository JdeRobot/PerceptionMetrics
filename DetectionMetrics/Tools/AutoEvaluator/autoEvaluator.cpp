#include <iostream>
#include <Utils/SampleGenerationApp.h>
#include <DatasetConverters/readers/OwnDatasetReader.h>
#include <FrameworkEvaluator/MassInferencer.h>
#include <DatasetConverters/readers/GenericDatasetReader.h>
#include <FrameworkEvaluator/GenericInferencer.h>
#include <FrameworkEvaluator/DetectionsEvaluator.h>
#include <FrameworkEvaluator/StatsWriter.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <glog/logging.h>
class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.push_back("Datasets.inputPath");
        this->requiredArguments.push_back("Datasets.readerImplementation");
        this->requiredArguments.push_back("Datasets.readerNames");
        this->requiredArguments.push_back("Inferencers.inferencerImplementation");
        this->requiredArguments.push_back("Inferencers.inferencerConfig");
        this->requiredArguments.push_back("Inferencers.inferencerWeights");
        this->requiredArguments.push_back("Inferencers.inferencerNames");
        this->requiredArguments.push_back("Inferencers.iouType");
        this->requiredArguments.push_back("outputCSVPath");


        /*this->requiredArguments.push_back("readerImplementationGT");
        this->requiredArguments.push_back("readerImplementationDetection");
        this->requiredArguments.push_back("readerNames");*/
    };

    void operator()(){
        YAML::Node datasetsNode=this->config.getNode("Datasets");
        YAML::Node inferencersNode=this->config.getNode("Inferencers");
        /*YAML::Node readerImplementationNode = this->config.getNode("readerImplementation");
        YAML::Node infererImplementationNode = this->config.getNode("inferencerImplementation");
        YAML::Node inferencerConfigNode = this->config.getNode("inferencerConfig");
        YAML::Node inferencerWeightsNode = this->config.getNode("inferencerWeights");
        YAML::Node inferencerNamesNode = this->config.getNode("inferencerNames");
        YAML::Node readerNamesNode = this->config.getNode("readerNames");
        */
        YAML::Node outputCSVNode = this->config.getNode("outputCSVPath");

        if (outputCSVNode.IsSequence())
            throw std::invalid_argument("Provided 'outputCSVPath' must be a single Directory, not multiple");

        auto boostPath= boost::filesystem::path(outputCSVNode.as<std::string>());
        if (boost::filesystem::exists(boostPath)) {
            if (!boost::filesystem::is_directory(boostPath)) {
                throw std::invalid_argument("Provided 'outputCSVPath' must be a Directory, not a file");
            }
        } else {
            boost::filesystem::create_directories(boostPath);
        }


        /*std::vector<std::string> inputPaths = inputPathsNode.IsSequence()
                                             ? inputPathsNode.as<std::vector<std::string>>()
                                             : std::vector<std::string>(1, inputPathsNode.as<std::string>());

        std::vector<std::string> inferencerWeights = inferencerWeightsNode.IsSequence()
                                                     ? inferencerWeightsNode.as<std::vector<std::string>>()
                                                     : std::vector<std::string>(1, inferencerWeightsNode.as<std::string>());

        */
        GenericDatasetReaderPtr reader;

        int count = 0;

        for (auto it = datasetsNode.begin(); it != datasetsNode.end(); it++) {


            if(!((*it)["inputPath"] && (*it)["readerNames"] && (*it)["readerImplementation"]))
                throw std::invalid_argument("Invalid Config file, Error Detected in Datasets Configuration");


            std::string inputPath = (*it)["inputPath"].as<std::string>();

            std::string readerNames = (*it)["readerNames"].as<std::string>();

            std::string readerImplementation = (*it)["readerImplementation"].as<std::string>();


            reader = GenericDatasetReaderPtr(
                        new GenericDatasetReader(inputPath, readerNames, readerImplementation, true));


            int count2 = 0;

            std::string mywriterFile(outputCSVNode.as<std::string>() + "/Dataset" + std::to_string(++count) + ".csv" );

            StatsWriter writer(reader->getReader(), mywriterFile);


            for (auto iter = inferencersNode.begin(); iter != inferencersNode.end(); iter++) {



                DatasetReaderPtr readerDetection ( new DatasetReader(true) );

                if(!((*iter)["inferencerConfig"] && (*iter)["inferencerNames"] && (*iter)["inferencerImplementation"]))
                    throw std::invalid_argument("Invalid Config file, Error Detected in Datasets Configuration");

                std::string inferencerConfig = (*iter)["inferencerConfig"].as<std::string>();

                std::string inferencerNames = (*iter)["inferencerNames"].as<std::string>();

                std::string inferencerWeights = (*iter)["inferencerWeights"].as<std::string>();

                std::string inferencerImplementation = (*iter)["inferencerImplementation"].as<std::string>();

                std::string inferencerIouType = (*iter)["iouType"].as<std::string>();

                bool isIouTypeBbox;

                if (inferencerIouType == "segm" || inferencerIouType == "bbox") {
                    isIouTypeBbox = inferencerIouType == "bbox";
                } else {
                    throw std::invalid_argument("Evaluation iouType can either be 'segm' or 'bbox'\n");
                }

                bool useDepth = (*iter)["useDepth"] ? (*iter)["useDepth"].as<bool>() : false;

                reader->getReader()->resetReaderCounter();

                GenericInferencerPtr inferencer(new GenericInferencer(inferencerConfig, inferencerWeights, inferencerNames, inferencerImplementation));
                MassInferencer massInferencer(reader->getReader(),inferencer->getInferencer(), false);
                massInferencer.process(useDepth, readerDetection);

                /*std::vector<Sample>::iterator iter;
                std::cout << samples.size() << '\n';

                    for(iter = samples.begin(); iter != samples.end(); iter++) {
                    RectRegionsPtr myrectregions = iter->getRectRegions();
                    std::vector<RectRegion> vec_regions = myrectregions->getRegions();

                    for (auto it = vec_regions.begin(), end= vec_regions.end(); it != end; ++it){
                      std::cout << "ClassID: " << it->classID.c_str() << '\n';

                    }

                }*/

                reader->getReader()->resetReaderCounter();

                //GenericDatasetReaderPtr readerGT(new GenericDatasetReader(inputPathGT.as<std::string>(),readerNamesNode.as<std::string>(), readerImplementationGTNode.as<std::string>()));

                DetectionsEvaluatorPtr evaluator(new DetectionsEvaluator(reader->getReader(),readerDetection,true));

                evaluator->evaluate(isIouTypeBbox);
                evaluator->accumulateResults();
                /*Extract weights name with folder*/
                std::string path = inferencerWeights;
                std::size_t a =  path.find_last_of("/");
                std::size_t b =  path.substr(0, a).find_last_of("/");
                a =  path.find_last_of(".");

                writer.writeInferencerResults(path.substr(b + 1, a - (b+1)), evaluator,massInferencer.getInferencer()->getMeanDurationTime());


                count2++;

            }

            writer.saveFile();

            count++;


        }

    };
};



int main (int argc, char* argv[]) {

    MyApp myApp(argc,argv);
    myApp.process();

    LOG(INFO) << "Auto Evaluation Successfull \n" ;
}
