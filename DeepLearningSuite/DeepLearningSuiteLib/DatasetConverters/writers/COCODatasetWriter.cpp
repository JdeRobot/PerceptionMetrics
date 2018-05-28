#include "COCODatasetWriter.h"
#include "DatasetConverters/ClassTypeOwn.h"
#include <iomanip>
#include <fstream>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <glog/logging.h>

using namespace rapidjson;

COCODatasetWriter::COCODatasetWriter(const std::string &outPath, DatasetReaderPtr &reader,bool overWriteclassWithZero):DatasetWriter(outPath,reader),overWriteclassWithZero(overWriteclassWithZero){

    this->fullImagesPath=boost::filesystem::absolute(boost::filesystem::path(outPath + "/train")).string();
    this->fullLabelsPath=boost::filesystem::absolute(boost::filesystem::path(outPath + "/annotations")).string();


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



    while (reader->getNextSample(sample)){
        auto boundingBoxes = sample.getRectRegions()->getRegions();
        std::string id = sample.getSampleID();

        std::stringstream ssID ;
        ssID << std::setfill('0') << std::setw(12 - id.length()) << id;
        std::string imageFileName = "COCO_train_" + ssID.str() + ".jpg";
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
        writer_imgs.Int(stoi(id));
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
                ClassTypeOwn typeConverter(it->classID);
                classId = typeConverter.getClassID();
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
            writer_anns.Int(stoi(id));
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

        LOG(INFO) << "Successfully Converted given Dataset to COCO dataset\n";
}
