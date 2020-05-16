#ifndef SAMPLERGENERATOR_STATSWRITER_H
#define SAMPLERGENERATOR_STATSWRITER_H

#include <map>
#include <FrameworkEvaluator/DetectionsEvaluator.h>
#include <DatasetConverters/readers/GenericDatasetReader.h>

class StatsWriter {
public:
    StatsWriter(DatasetReaderPtr dataset, std::string& writerFile);
    void writeInferencerResults(std::string inferencerName, DetectionsEvaluatorPtr evaluator, unsigned int mean_inference_time = 0);
    void saveFile();

private:
    std::ofstream writer;
    std::string writerFile;
    std::vector<std::string> classNamesinOrder;
};


#endif //SAMPLERGENERATOR_STATSWRITER_H
