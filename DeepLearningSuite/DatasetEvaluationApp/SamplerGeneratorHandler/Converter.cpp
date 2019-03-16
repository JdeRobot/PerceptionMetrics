//
// Created by frivas on 19/02/17.
//

#include <DatasetConverters/writers/GenericDatasetWriter.h>
#include <gui/Utils.h>
#include "Converter.h"
#include "SamplerGenerationHandler.h"
#include <glog/logging.h>
void SampleGeneratorHandler::Converter::process(QListView *datasetList, QListView *namesList, QListView *readerImpList,
                                                QListView *filterClasses, QListView *writerImpList, QListView* writerNamesList, bool useWriterNames,
                                                const std::string& datasetPath, const std::string& namesPath, const std::string &outputPath,
                                                bool splitActive, double splitRatio,bool writeImages) {


    GenericDatasetReaderPtr reader = SamplerGenerationHandler::createDatasetReaderPtr(datasetList, namesList,
                                                                                      readerImpList, filterClasses,
                                                                                      datasetPath, namesPath, writeImages);   // Images Required for dataset if and
                                                                                                                              // only if write images is true


    if (!reader)
        return;


    std::vector<std::string> writerImp;
    Utils::getListViewContent(writerImpList,writerImp,"");
    std::vector<std::string> writerNames;

    if (useWriterNames) {
        if (! Utils::getListViewContent(writerNamesList,writerNames,namesPath+"/")){
            LOG(WARNING)<<"Select the dataset names related to the Output dataset, or unchechk mapping if you want a custom names file to be generated";
            return;
    }


}


    if (splitActive){
        DatasetReaderPtr readerTest(new DatasetReader(writeImages));
        DatasetReaderPtr readerTrain(new DatasetReader(writeImages));

        std::string testPath = outputPath + "/test";
        std::string trainPath = outputPath + "/train";


        int ratio=int(splitRatio*10);

        Sample sample;
        auto readerPtr = reader->getReader();
        int counter=0;
        while (readerPtr->getNextSample(sample)){
            if (counter <ratio){
                readerTrain->addSample(sample);
            }
            else{
                readerTest->addSample(sample);
            }
            counter++;
            counter= counter % 10;
        }

        LOG(INFO) << "Train: " << std::endl;
        readerTrain->printDatasetStats();
        LOG(INFO) << "Test: " << std::endl;
        readerTest->printDatasetStats();

        if (useWriterNames) {
            GenericDatasetWriterPtr writerTest( new GenericDatasetWriter(testPath,readerTest,writerImp[0],writerNames[0]));
            GenericDatasetWriterPtr writerTrain( new GenericDatasetWriter(trainPath,readerTrain,writerImp[0], writerNames[0]));
            writerTest->getWriter()->process(writeImages);
            writerTrain->getWriter()->process(writeImages);

        } else {
            GenericDatasetWriterPtr writerTest( new GenericDatasetWriter(testPath,readerTest,writerImp[0]));
            GenericDatasetWriterPtr writerTrain( new GenericDatasetWriter(trainPath,readerTrain,writerImp[0]));
            writerTest->getWriter()->process(writeImages);
            writerTrain->getWriter()->process(writeImages);

        }

    }
    else{
        auto readerPtr = reader->getReader();
        if (useWriterNames) {
            GenericDatasetWriterPtr writer( new GenericDatasetWriter(outputPath,readerPtr,writerImp[0], writerNames[0]));

            writer->getWriter()->process(writeImages);
        } else {
            GenericDatasetWriterPtr writer( new GenericDatasetWriter(outputPath,readerPtr,writerImp[0]));

            writer->getWriter()->process(writeImages);
        }

    }


}
