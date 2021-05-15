//
// Created by frivas on 19/02/17.
//

#ifndef SAMPLERGENERATOR_CONVERTER_H
#define SAMPLERGENERATOR_CONVERTER_H

#include <QtWidgets/QListView>

namespace SampleGeneratorHandler {
    class Converter {
    public:
        static void process(QListView* datasetList,QListView* namesList,QListView* readerImpList, QListView* filterClasses, QListView* writerImpList,
                            QListView* writerNamesList, bool useWriterNames, const std::string& datasetPath, const std::string& namesPath, const std::string& outputPath, bool splitActive, double splitRatio, bool writeImages);
    };

}


#endif //SAMPLERGENERATOR_CONVERTER_H
