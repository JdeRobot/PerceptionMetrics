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

class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.push_back("inputPath");
        this->requiredArguments.push_back("readerImplementation");
        this->requiredArguments.push_back("inferencerImplementation");
        this->requiredArguments.push_back("inferencerConfig");
        this->requiredArguments.push_back("inferencerWeights");
        this->requiredArguments.push_back("inferencerNames");
        this->requiredArguments.push_back("readerNames");
        this->requiredArguments.push_back("outputCSVPath");


        /*this->requiredArguments.push_back("readerImplementationGT");
        this->requiredArguments.push_back("readerImplementationDetection");
        this->requiredArguments.push_back("readerNames");*/
    };

    void operator()(){
        Key inputPathsKey=this->config->getKey("inputPath");
        Key readerImplementationKey = this->config->getKey("readerImplementation");
        Key infererImplementationKey = this->config->getKey("inferencerImplementation");
        Key inferencerConfigKey = this->config->getKey("inferencerConfig");
        Key inferencerWeightsKey = this->config->getKey("inferencerWeights");
        Key inferencerNamesKey = this->config->getKey("inferencerNames");
        Key readerNamesKey = this->config->getKey("readerNames");
        Key outputCSVKey = this->config->getKey("outputCSVPath");

        if (outputCSVKey.isVector())
            throw std::invalid_argument("Provided 'outputCSVPath' must be a single Directory, not multiple");

        auto boostPath= boost::filesystem::path(outputCSVKey.getValue());
        if (!boost::filesystem::is_directory(boostPath)) {
            throw std::invalid_argument("Provided 'outputCSVPath' must be a Directory, not a file");
        }
        if (!boost::filesystem::exists(boostPath)){
            boost::filesystem::create_directories(boostPath);
        }


        std::vector<std::string> inputPaths = inputPathsKey.getValues();
        std::vector<std::string> inferencerWeights = inferencerWeightsKey.getValues();


        GenericDatasetReaderPtr reader;

        int count = 0;

        for (auto it = inputPaths.begin(); it != inputPaths.end(); it++) {


            std::string readerNames = readerNamesKey.getValueOrLast(count);
            std::string readerImplementation = readerImplementationKey.getValueOrLast(count);

            reader = GenericDatasetReaderPtr(
                        new GenericDatasetReader(*it, readerNames, readerImplementation));


            int count2 = 0;

            std::string mywriterFile(outputCSVKey.getValue() + "/Dataset" + std::to_string(++count) + ".csv" );

            StatsWriter writer(reader->getReader(), mywriterFile);


            for (auto iter = inferencerWeights.begin(); iter != inferencerWeights.end(); iter++) {



                std::vector<Sample> samples;
                std::string inferencerConfig = inferencerConfigKey.getValueOrLast(count2);
                std::string inferencerNames = inferencerNamesKey.getValueOrLast(count2);
                std::string inferencerImplementation = infererImplementationKey.getValueOrLast(count2);


                reader->getReader()->resetReaderCounter();

                GenericInferencerPtr inferencer(new GenericInferencer(inferencerConfig, *iter, inferencerNames,inferencerImplementation));
                MassInferencer massInferencer(reader->getReader(),inferencer->getInferencer(), true);
                massInferencer.process(false, &samples);

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

                //GenericDatasetReaderPtr readerGT(new GenericDatasetReader(inputPathGT.getValue(),readerNamesKey.getValue(), readerImplementationGTKey.getValue()));
                GenericDatasetReaderPtr readerDetection(new GenericDatasetReader(samples, inferencerNamesKey.getValue()));


                DetectionsEvaluatorPtr evaluator(new DetectionsEvaluator(reader->getReader(),readerDetection->getReader(),true));

                evaluator->evaluate();

                /*Extract weights name with folder*/
                std::string path = *iter;
                std::size_t a =  path.find_last_of("/");
                std::size_t b =  path.substr(0, a).find_last_of("/");
                a =  path.find_last_of(".");

                writer.writeInferencerResults(path.substr(b + 1, a - (b+1)), evaluator->getStats());


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

    std::cout << "Auto Evaluation Successfull" << '\n';
}
