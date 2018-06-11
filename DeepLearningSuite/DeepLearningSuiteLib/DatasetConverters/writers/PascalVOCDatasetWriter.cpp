#include "PascalVOCDatasetWriter.h"
#include "DatasetConverters/ClassTypeMapper.h"
#include <iomanip>
#include <fstream>
#include <glog/logging.h>


PascalVOCDatasetWriter::PascalVOCDatasetWriter(const std::string &outPath, DatasetReaderPtr &reader,const std::string& writerNamesFile,bool overWriteclassWithZero):DatasetWriter(outPath,reader),writerNamesFile(writerNamesFile),overWriteclassWithZero(overWriteclassWithZero){

    this->fullImagesPath=boost::filesystem::absolute(boost::filesystem::path(outPath + "/VOCDevKit/VOC20xx/JPEGImages")).string();
    this->fullLabelsPath=boost::filesystem::absolute(boost::filesystem::path(outPath + "/VOCDevKit/VOC20xx/Annotations")).string();
    this->fullNamesPath=boost::filesystem::absolute(boost::filesystem::path(outPath + "/voc.names")).string();

    auto boostImages= boost::filesystem::path(fullImagesPath);
    if (!boost::filesystem::exists(boostImages)){
        boost::filesystem::create_directories(boostImages);
    }
    auto boostLabels= boost::filesystem::path(fullLabelsPath);
    if (!boost::filesystem::exists(boostLabels)){
        boost::filesystem::create_directories(boostLabels);
    }


    std::cout << "FullImagesPath: " << this->fullImagesPath << std::endl;
    std::cout << "FullLabelsPath: " << this->fullImagesPath << std::endl;

}

void PascalVOCDatasetWriter::process(bool usedColorImage) {

    Sample sample;

    ClassTypeMapper typeMapper;

    if (!writerNamesFile.empty())
        typeMapper = ClassTypeMapper(writerNamesFile);

    int count = 0;

    while (reader->getNextSample(sample)){
        count++;
        if (count == 5000)
            break;

        auto boundingBoxes = sample.getRectRegions()->getRegions();
        std::string id = sample.getSampleID();

        std::string imageFilePath= this->fullImagesPath + "/" + id + ".jpg";

        std::string labelFilePath= this->fullLabelsPath + "/" + id + ".xml";

        cv::Mat image;
        if (usedColorImage)
            image= sample.getColorImage();
        else {
            image = sample.getDepthColorMapImage();

        }

        boost::property_tree::ptree tree;

        tree.put("annotation.filename", id + ".jpg");
        tree.put("annotation.folder", "VOC20xx");


        for (auto it = boundingBoxes.begin(), end=boundingBoxes.end(); it != end; ++it){
            double x = it->region.x;
            double y = it->region.y;
            double w = it->region.width;
            double h = it->region.height;

            std::string className;

            if (overWriteclassWithZero)
              className = "all";
            else {
              if (writerNamesFile.empty()) {
                  if (find(this->outputClasses.begin(), this->outputClasses.end(), it->classID) == this->outputClasses.end())
                      this->outputClasses.push_back(it->classID);
              } else {
                  if(typeMapper.mapString(it->classID)) {         // Mapping Successfull
                      className = typeMapper.getClassString();

                      if (it->classID != className)
                          this->mapped_classes[it->classID] = className;

                  } else {                                        // No Mapping Found Discarding Class
                      std::unordered_map<std::string, long int>::iterator itr = this->discarded_classes.find(it->classID);
                      if (itr != this->discarded_classes.end()) {
                          itr->second++;
                      } else {
                          this->discarded_classes.insert(std::make_pair(it->classID, 1));
                      }
                      continue;
                  }

              }

            }

            boost::property_tree::ptree & node = tree.add("annotation.object", "");

            node.put("name", className);
            node.put("bndbox.xmin", x);
            node.put("bndbox.xmax", x + w);
            node.put("bndbox.ymin", y);
            node.put("bndbox.ymax", y + h);

        }

        tree.add("annotation.size.depth", 3);
        tree.add("annotation.size.height", image.size().height);
        tree.add("annotation.size.width", image.size().width);

        boost::property_tree::write_xml(labelFilePath, tree);

        cv::imwrite(imageFilePath,image);
    }

    if (writerNamesFile.empty()) {

        std::ofstream writerClassfile;
        writerClassfile.open (this->fullNamesPath);

        std::vector<std::string>::iterator it;
        for (it = this->outputClasses.begin(); it != this->outputClasses.end(); it++) {
            writerClassfile << *it << "\n";
        }
        writerClassfile.close();
    }


    LOG(INFO) << "Successfully Converted given Dataset to Pascal VOC dataset\n";

    if (writerNamesFile.empty()) {

        std::cout << "\nPrinting Mapping Info\n";
        std::cout << "**********************\n";

        for (std::unordered_map<std::string, std::string>::iterator it=this->mapped_classes.begin(); it!=this->mapped_classes.end(); ++it)
            std::cout << it->first << " => " << it->second << '\n';

        std::cout << "**********************\n";

        std::cout << "\nPrinting Discarded Classes from Original Dataset\n";
        std::cout << "**********************\n";

        for (std::unordered_map<std::string, long int>::iterator it=this->discarded_classes.begin(); it!=this->discarded_classes.end(); ++it)
            std::cout << it->first << " : " << it->second << '\n';
        std::cout << "**********************\n";

    }

}
