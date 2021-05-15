#include "StatsWriter.h"
#include <glog/logging.h>
StatsWriter::StatsWriter(DatasetReaderPtr dataset, std::string& writerFile) {

    this->writerFile = writerFile;
    this->writer = std::ofstream(writerFile);

    std::ifstream classNamesReader(dataset->getClassNamesFile());

    std::string className;

    int counter = 0;
    while(getline(classNamesReader, className)) {

	    if (className.empty())
            continue;
        if (counter == 0)
            this->writer << ", " << className;
        else
            this->writer << ",," << className;
        counter++;
        this->classNamesinOrder.push_back(className);
    }

    this->writer << "\n";

    for (int i = 0; i< counter; i++) {
        this->writer << ", mAP(IOU=0.5:0.95), mAR(IOU=0.5:0.95)";
    }

    this->writer << ", mAP(Overall)(IOU=0.5:0.95), mAR(Overall)(IOU=0.5:0.95)";
    this->writer << " , Mean inference time(ms) , Time Taken in Evaluation (second), Time Taken in Accumulation (second)";

    this->writer << "\n";

}

void StatsWriter::writeInferencerResults(std::string inferencerName, DetectionsEvaluatorPtr evaluator, unsigned int mean_inference_time) {

    this->writer << inferencerName;
    std::map<std::string, double> meanAP = evaluator->getClassWiseAP();
    std::map<std::string, double> meanAR = evaluator->getClassWiseAR();

    std::map<std::string,ClassStatistics>::const_iterator iter;

    for (std::vector<std::string>::iterator it = this->classNamesinOrder.begin(); it != this->classNamesinOrder.end(); it++) {
        if ((*it).empty())
            continue;
        if (meanAP.count(*it)) {
            double AP = meanAP.at(*it);
            double AR = meanAR.at(*it);
            this->writer << ", " << AP << ", " << AR;
        } else {
            LOG(INFO) << "Class " << *it << " not present!!" << " Skipping";
            this->writer << ",,";
        }

    }
      this->writer << ", " << evaluator->getOverallmAP() << ", " << evaluator->getOverallmAR();
      this->writer << ", " << mean_inference_time << ", "<< evaluator->getEvaluationTime() << ", " << evaluator->getAccumulationTime();
      this->writer << "\n";
      LOG(INFO) << "Inference Results Written Successfully";

      this->writer.flush(); // Update File contents

}

void StatsWriter::saveFile() {
    this->writer.close();
    LOG(INFO) << "File " << this->writerFile << " Saved Successfully" << '\n';
}
