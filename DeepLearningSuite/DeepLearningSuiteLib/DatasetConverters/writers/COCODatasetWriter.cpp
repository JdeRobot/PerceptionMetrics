#include "COCODatasetWriter.h"
#include "DatasetConverters/ClassTypeMapper.h"
#include <iomanip>
#include <math.h>
#include <fstream>
#include <algorithm>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <glog/logging.h>

using namespace rapidjson;

COCODatasetWriter::COCODatasetWriter(const std::string &outPath, DatasetReaderPtr &reader, const std::string& writerNamesFile, bool overWriteclassWithZero):DatasetWriter(outPath,reader),overWriteclassWithZero(overWriteclassWithZero), writerNamesFile(writerNamesFile){

    this->fullImagesPath=boost::filesystem::absolute(boost::filesystem::path(outPath + "/train")).string();
    this->fullLabelsPath=boost::filesystem::absolute(boost::filesystem::path(outPath + "/annotations")).string();
    this->fullNamesPath=boost::filesystem::absolute(boost::filesystem::path(outPath + "/coco.names")).string();

    auto boostImages= boost::filesystem::path(fullImagesPath);
    if (!boost::filesystem::exists(boostImages)){
        boost::filesystem::create_directories(boostImages);
    }
    auto boostLabels= boost::filesystem::path(fullLabelsPath);
    if (!boost::filesystem::exists(boostLabels)){
        boost::filesystem::create_directories(boostLabels);
    }


    LOG(INFO) << "FullImagesPath: " << this->fullImagesPath << std::endl;
    LOG(INFO) << "FullLabelsPath: " << this->fullLabelsPath << std::endl;

}

void COCODatasetWriter::process(bool writeImages, bool useDepth) {
    Sample sample;
    //std::ofstream sampleFile(this->outPath + "/sample.txt");

    StringBuffer s_anns;
    StringBuffer s_imgs;

    std::string labelFilePath= this->fullLabelsPath + "/" + "instances_train.json";
    std::ofstream out(labelFilePath);

    Writer<StringBuffer> writer_anns(s_anns);
    Writer<StringBuffer> writer_imgs(s_imgs);

    writer_anns.StartObject();
    writer_anns.Key("annotations");
    writer_anns.StartArray();

    if (writeImages) {
        writer_imgs.StartObject();
        writer_imgs.Key("images");
        writer_imgs.StartArray();

    }

    ClassTypeMapper typeMapper;

    if (!writerNamesFile.empty())
        typeMapper = ClassTypeMapper(writerNamesFile);

    int id = 0;

    while (reader->getNextSample(sample)){

        auto boundingBoxes = sample.getRectRegions()->getRegions();
        auto segmentationRegions = sample.getRleRegions()->getRegions();
        int width = sample.getSampleWidth();
        int height = sample.getSampleHeight();

        std::string id_string = sample.getSampleID();
        id++;
        id_string.erase(std::remove_if(id_string.begin(), id_string.end(), isspace), id_string.end());


        std::string::size_type sz;   // alias of size_t

        int num_id = std::stoi (id_string, &sz);

        std::string imageFileName;
        if (id_string.length() == sz) {
            int i = ceil(log10(num_id));

            std::string ssID (12-i+1,'0') ;

            imageFileName = "COCO_" + ssID + std::to_string(num_id) + ".jpg";
        } else {
            imageFileName = "COCO_" + std::to_string(id) + ".jpg";
            num_id = id;
        }


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
                cv::imwrite(this->fullImagesPath + "/" + imageFileName,image);

            }

        }

        if (writeImages) {
            writer_imgs.StartObject();
            writer_imgs.Key("file_name");
            writer_imgs.String(imageFileName.c_str());
            writer_imgs.Key("id");
            writer_imgs.Int(num_id);
            writer_imgs.Key("height");
            writer_imgs.Int(height);
            writer_imgs.Key("width");
            writer_imgs.Int(width);
            writer_imgs.EndObject();

        }


        int i = 0;
        for (auto it = boundingBoxes.begin(), end=boundingBoxes.end(); it != end; ++it){

            double x = it->region.x;
            double y = it->region.y;
            double w = it->region.width;
            double h = it->region.height;
            double confidence_score = it->confidence_score;

            int classId;
            if (overWriteclassWithZero)
                classId=0;
            else {
                if (writerNamesFile.empty()) {
                    std::vector<std::string>::iterator itr;
                    itr = find(this->outputClasses.begin(), this->outputClasses.end(), it->classID);
                    if (itr == this->outputClasses.end()) {
                        this->outputClasses.push_back(it->classID);
                        classId = this->outputClasses.size() - 1;
                    } else {
                        classId = std::distance(this->outputClasses.begin(), itr);
                    }

                } else {
                    if(typeMapper.mapString(it->classID)) {         // Mapping Successfull
                        classId = typeMapper.getClassID();
                        if (it->classID != typeMapper.getClassString())
                            this->mapped_classes[it->classID] = typeMapper.getClassString();

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
                //ClassTypeMapper typeMapper(it->classID);
                //classId = typeConverter.getClassID();
            }

            writer_anns.StartObject();
            writer_anns.Key("bbox");
            writer_anns.StartArray();
            writer_anns.Double(x);
            writer_anns.Double(y);
            writer_anns.Double(w);
            writer_anns.Double(h);
            writer_anns.EndArray();
            writer_anns.Key("category_id");
            writer_anns.Int(classId + 1);        // Classes in DetectionSuite start
                                                // 0 wherease it starts from 1 in
                                                // COCO dataset, that's why +
            writer_anns.Key("score");
            writer_anns.Double(confidence_score);
            writer_anns.Key("image_id");
            writer_anns.Int(num_id);

            if (!segmentationRegions.empty()) {
                writer_anns.Key("segmentation");
                writer_anns.StartObject();
                writer_anns.Key("size");
                writer_anns.StartArray();
                writer_anns.Int(height);
                writer_anns.Int(width);
                writer_anns.EndArray();
                writer_anns.Key("counts");
                writer_anns.String(rleToString(&(segmentationRegions[i].region)));
                writer_anns.EndObject();

            }

            writer_anns.EndObject();
            i++;

        }


    }

        writer_anns.EndArray();
        writer_anns.EndObject();

        std::string json_anns (s_anns.GetString(), s_anns.GetSize());

        if (writeImages) {

            writer_imgs.EndArray();
            writer_imgs.EndObject();

            std::string json_imgs (s_imgs.GetString(), s_imgs.GetSize());


            json_imgs.pop_back();

            out << json_imgs;
            json_anns.erase(0, 1);
            out << ",";
        }

        out << json_anns;


        if (!out.good()) throw std::runtime_error ("Can't write the JSON string to the file!");


        if (writerNamesFile.empty()) {

            std::ofstream writerClassfile;
            writerClassfile.open (this->fullNamesPath);

            std::vector<std::string>::iterator it;
            for (it = this->outputClasses.begin(); it != this->outputClasses.end(); it++) {
                writerClassfile << *it << "\n";
            }
            writerClassfile.close();
        }

        LOG(INFO) << "Successfully Written to COCO dataset\n";

        if (!writerNamesFile.empty()) {

            LOG(INFO) << "\nPrinting Mapping Info\n";
            LOG(INFO) << "**********************\n";

            for (std::unordered_map<std::string, std::string>::iterator it=this->mapped_classes.begin(); it!=this->mapped_classes.end(); ++it)
                LOG(INFO) << it->first << " => " << it->second << '\n';

            LOG(INFO) << "**********************\n";

            LOG(WARNING) << "\nPrinting Discarded Classes from Original Dataset\n";
            LOG(INFO) << "**********************\n";

            for (std::unordered_map<std::string, long int>::iterator it=this->discarded_classes.begin(); it!=this->discarded_classes.end(); ++it)
                LOG(WARNING) << it->first << " : " << it->second << '\n';
            LOG(INFO) << "**********************\n";

        }
}
