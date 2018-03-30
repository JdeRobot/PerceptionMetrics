//
// Created by frivas on 18/02/17.
//

#ifndef SAMPLERGENERATOR_VIEWER_H
#define SAMPLERGENERATOR_VIEWER_H

#include <QtWidgets/QListView>

namespace SampleGeneratorHandler {
    class Viewer {
    public:
        static void process(QListView* datasetList,QListView* namesList,QListView* readerImpList, QListView* filterClasses, bool showDepth, const std::string& datasetPath, const std::string& namesPath);
    };

}

#endif //SAMPLERGENERATOR_VIEWER_H
