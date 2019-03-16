#include "PascalVOCDatasetWriter.h"
#include "DatasetConverters/ClassTypeMapper.h"
#include <iomanip>
#include <fstream>
#include <glog/logging.h>


PascalVOCDatasetWriter::PascalVOCDatasetWriter(const std::string &outPath, DatasetReaderPtr &reader,const std::string& writerNamesFile, bool overWriteclassWithZero):DatasetWriter(outPath,reader),writerNamesFile(writerNamesFile),overWriteclassWithZero(overWriteclassWithZero){

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


    LOG(INFO) << "FullImagesPath: " << this->fullImagesPath << std::endl;
    LOG(INFO) << "FullLabelsPath: " << this->fullImagesPath << std::endl;

}

void PascalVOCDatasetWriter::process(bool writeImages, bool useDepth) {

    Sample sample;

    ClassTypeMapper typeMapper;

    if (!writerNamesFile.empty())
        typeMapper = ClassTypeMapper(writerNamesFile);

    int count = 0;
    int skip_count = 0;

    while (reader->getNextSample(sample)){
        count++;
        if (count == 5000)
            break;

        auto boundingBoxes = sample.getRectRegions()->getRegions();
        std::string id = sample.getSampleID();

        std::string imageFilePath= this->fullImagesPath + "/" + id + ".jpg";
        std::string labelFilePath= this->fullLabelsPath + "/" + id + ".xml";

        cv::Mat image;
        if (writeImages) {
            if (useDepth) {
                image= sample.getDepthImage();
            } else {
                image= sample.getColorImage();
            }

            if (image.empty()) {
                skip_count++;
                if (skip_count > this->skip_count) {
                    throw std::runtime_error("Maximum limit for skipping exceeded, either turn off writing images or fix issues in dataset");
                }
                LOG(WARNING) << "Image empty, skipping writing image. Skipped " + std::to_string(skip_count) + " of " + std::to_string(this->skip_count);

            } else {
                cv::imwrite(imageFilePath,image);

            }

        }

        boost::property_tree::ptree tree;

        tree.put("annotation.filename", id + ".jpg");
        tree.put("annotation.folder", "VOC20xx");


        for (auto it = boundingBoxes.begin(), end=boundingBoxes.end(); it != end; ++it){
            double x = it->region.x;
            double y = it->region.y;
            double w = it->region.width;
            double h = it->region.height;

            double confidence_score = it->confidence_score;

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
            node.put("score", confidence_score);

        }

        tree.add("annotation.size.depth", 3);
        tree.add("annotation.size.height", image.size().height);
        tree.add("annotation.size.width", image.size().width);

        boost::property_tree::write_xml(labelFilePath, tree);


    }

    if (!writerNamesFile.empty()) {

        std::ofstream writerClassfile;
        writerClassfile.open (this->fullNamesPath);

        std::vector<std::string>::iterator it;
        for (it = this->outputClasses.begin(); it != this->outputClasses.end(); it++) {
            writerClassfile << *it << "\n";
        }
        writerClassfile.close();
    }


    LOG(INFO) << "Successfully Converted given Dataset to Pascal VOC dataset\n";

    if (!writerNamesFile.empty()) {

        LOG(INFO) << "\nPrinting Mapping Info\n";
        LOG(INFO) << "**********************\n";

        for (std::unordered_map<std::string, std::string>::iterator it=this->mapped_classes.begin(); it!=this->mapped_classes.end(); ++it)
            LOG(INFO) << it->first << " => " << it->second << '\n';

        LOG(INFO) << "**********************\n";

        LOG(INFO) << "\nPrinting Discarded Classes from Original Dataset\n";
        LOG(INFO) << "**********************\n";

        for (std::unordered_map<std::string, long int>::iterator it=this->discarded_classes.begin(); it!=this->discarded_classes.end(); ++it)
            LOG(INFO) << it->first << " : " << it->second << '\n';
        LOG(INFO) << "**********************\n";

    }

}
