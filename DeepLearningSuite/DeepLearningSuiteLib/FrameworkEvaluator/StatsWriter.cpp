#include "StatsWriter.h"

StatsWriter::StatsWriter(DatasetReaderPtr dataset, std::string& writerFile) {

    this->writerFile = writerFile;
    this->writer = std::ofstream(writerFile);

    std::ifstream classNamesReader(dataset->getClassNamesFile());

    std::string className;

    int counter = 0;
    while(getline(classNamesReader, className)) {
        if (counter == 0)
            this->writer << ", " << className;
        else
            this->writer << ",,, " << className;
        counter++;
        this->classNamesinOrder.push_back(className);
    }

    this->writer << "\n";

    for (int i = 0; i< counter; i++) {
        this->writer << ", Mean IOU, Precision, Recall";
    }

    this->writer << "\n";

}

void StatsWriter::writeInferencerResults(std::string inferencerName, GlobalStats stats) {

    this->writer << inferencerName;
    std::map<std::string,ClassStatistics> results = stats.getStats();

    std::map<std::string,ClassStatistics>::const_iterator iter;

    for (std::vector<std::string>::iterator it = this->classNamesinOrder.begin(); it != this->classNamesinOrder.end(); it++) {
        if ((*it).empty())
            continue;
        if (results.count(*it)) {
            auto data = results.at(*it);
            this->writer << ", " << data.getMeanIOU() << ", " << data.getPrecision() << ", " << data.getRecall();
        } else {
            std::cout << "Class " << *it << " not present!!" << " Skipping" << '\n';
            this->writer << ",,,";
        }

    }

      this->writer << "\n";
      std::cout << "Inference Results Written Successfully" << '\n';

}

void StatsWriter::saveFile() {
    this->writer.close();
    std::cout << "File " << this->writerFile << " Saved Successfully" << '\n';
}
