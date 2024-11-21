//
// Created by frivas on 20/02/17.
//

#ifndef SAMPLERGENERATOR_EVALUATOR_H
#define SAMPLERGENERATOR_EVALUATOR_H


#include <QtWidgets/QListView>


namespace SampleGeneratorHandler {
    class Evaluator {
    public:
        static void process(QListView* datasetListGT,QListView* namesListGT,QListView* readerImpListGT,
                            QListView* datasetListDetect,QListView* namesListDetect,QListView* readerImpListDetect,
                            QListView* filterClasses, const std::string& datasetPath, const std::string& namesGTPath,
                            const std::string& inferencesPath, const std::string& namesPath,bool overWriterPersonClasses,
                            bool enableMixEvaluation, bool isIouTypeBbox
        );
    };

}

#endif //SAMPLERGENERATOR_EVALUATOR_H
