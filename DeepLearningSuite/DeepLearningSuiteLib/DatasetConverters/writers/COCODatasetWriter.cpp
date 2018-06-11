#include "COCODatasetWriter.h"
#include "DatasetConverters/ClassTypeMapper.h"
#include <iomanip>
#include <fstream>
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


    std::cout << "FullImagesPath: " << this->fullImagesPath << std::endl;
    std::cout << "FullLabelsPath: " << this->fullLabelsPath << std::endl;

}

void COCODatasetWriter::process(bool usedColorImage) {
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


    writer_imgs.StartObject();
    writer_imgs.Key("images");
    writer_imgs.StartArray();

    ClassTypeMapper typeMapper;

    if (!writerNamesFile.empty())
        typeMapper = ClassTypeMapper(writerNamesFile);

    int id = 0;

    while (reader->getNextSample(sample)){
        auto boundingBoxes = sample.getRectRegions()->getRegions();
        //std::string id_string = sample.getSampleID();

        if (id == 100)
            break;

        id++;

        int i;
        for(int num = id, i=0; num > 0; num=num/10, i++);

        std::stringstream ssID ;
        ssID << std::setfill('0') << std::setw(12 - i) << id;
        std::string imageFileName = "COCO_" + ssID.str() + ".jpg";
        std::string imageFilePath= this->fullImagesPath + "/" + imageFileName;

        cv::Mat image;
        if (usedColorImage)
            image= sample.getColorImage();
        else {
            image = sample.getDepthColorMapImage();

        }

        writer_imgs.StartObject();
        writer_imgs.Key("file_name");
        writer_imgs.String(imageFileName.c_str());
        writer_imgs.Key("id");
        writer_imgs.Int(id);
        writer_imgs.Key("height");
        writer_imgs.Int(image.size().height);
        writer_imgs.Key("width");
        writer_imgs.Int(image.size().width);
        writer_imgs.EndObject();


        for (auto it = boundingBoxes.begin(), end=boundingBoxes.end(); it != end; ++it){

            double x = it->region.x;
            double y = it->region.y;
            double w = it->region.width;
            double h = it->region.height;

            int classId;
            if (overWriteclassWithZero)
                classId=0;
            else {
                if (writerNamesFile.empty()) {
                    std::vector<std::string>::iterator itr;
                    itr = find(this->outputClasses.begin(), this->outputClasses.end(), it->classID);
                    if (itr == this->outputClasses.end()) {
                        this->outputClasses.push_back(it->classID);
                        classId = std::distance(this->outputClasses.begin(), itr) + 1;
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
            writer_anns.Key("image_id");
            writer_anns.Int(id);
            writer_anns.EndObject();


        }

        cv::imwrite(imageFilePath,image);

    }
        writer_anns.EndArray();
        writer_anns.EndObject();

        writer_imgs.EndArray();
        writer_imgs.EndObject();

        std::string json_anns (s_anns.GetString(), s_anns.GetSize());
        std::string json_imgs (s_imgs.GetString(), s_imgs.GetSize());

        std::size_t pos = json_anns.find_last_of("}");
        json_anns.erase(pos, 1);
        out << json_anns;
        out << ",";

        pos = json_imgs.find_first_of("{");
        json_imgs.erase(pos, 1);

        out << json_imgs;

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

        LOG(INFO) << "Successfully Converted given Dataset to COCO dataset\n";

        if (!writerNamesFile.empty()) {

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
