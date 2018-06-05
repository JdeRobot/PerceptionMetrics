#include <iostream>
#include <Utils/SampleGenerationApp.h>
#include <DatasetConverters/readers/OwnDatasetReader.h>
#include <FrameworkEvaluator/MassInferencer.h>
#include <DatasetConverters/readers/GenericDatasetReader.h>
#include <FrameworkEvaluator/GenericInferencer.h>
#include <FrameworkEvaluator/DetectionsEvaluator.h>

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


        /*this->requiredArguments.push_back("readerImplementationGT");
        this->requiredArguments.push_back("readerImplementationDetection");
        this->requiredArguments.push_back("readerNames");*/
    };
    void operator()(){
        Key inputPath=this->config->getKey("inputPath");
        Key outputPath=this->config->getKey("outputPath");

        Key readerImplementationKey = this->config->getKey("readerImplementation");
        Key infererImplementationKey = this->config->getKey("inferencerImplementation");
        Key inferencerConfigKey = this->config->getKey("inferencerConfig");
        Key inferencerWeightsKey = this->config->getKey("inferencerWeights");
        Key inferencerNamesKey = this->config->getKey("inferencerNames");
        Key readerNamesKey = this->config->getKey("readerNames");

        GenericDatasetReaderPtr reader;
        if (inputPath.isVector()) {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPath.getValues(),readerNamesKey.getValue(), readerImplementationKey.getValue()));
        }
        else {
            reader = GenericDatasetReaderPtr(
                    new GenericDatasetReader(inputPath.getValue(),readerNamesKey.getValue(), readerImplementationKey.getValue()));
        }

        std::vector<Sample> samples;

        GenericInferencerPtr inferencer(new GenericInferencer(inferencerConfigKey.getValue(),inferencerWeightsKey.getValue(),inferencerNamesKey.getValue(),infererImplementationKey.getValue()));
        MassInferencer massInferencer(reader->getReader(),inferencer->getInferencer(),outputPath.getValue(), true);
        massInferencer.process(false, &samples);

        std::vector<Sample>::iterator iter;
        std::cout << samples.size() << '\n';

            for(iter = samples.begin(); iter != samples.end(); iter++) {
            RectRegionsPtr myrectregions = iter->getRectRegions();
            std::vector<RectRegion> vec_regions = myrectregions->getRegions();

            for (auto it = vec_regions.begin(), end= vec_regions.end(); it != end; ++it){
              std::cout << "ClassID: " << it->classID.c_str() << '\n';

            }

        }

        reader->getReader()->resetReaderCounter();

        //GenericDatasetReaderPtr readerGT(new GenericDatasetReader(inputPathGT.getValue(),readerNamesKey.getValue(), readerImplementationGTKey.getValue()));
        GenericDatasetReaderPtr readerDetection(new GenericDatasetReader(samples));


        DetectionsEvaluatorPtr evaluator(new DetectionsEvaluator(reader->getReader(),readerDetection->getReader(),true));

        evaluator->evaluate();

    };
};



int main (int argc, char* argv[]) {

    MyApp myApp(argc,argv);
    myApp.process();
}
