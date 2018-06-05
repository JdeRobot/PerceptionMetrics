#include <fstream>
#include <glog/logging.h>
#include <boost/filesystem/path.hpp>
#include "COCODatasetReader.h"
#include "DatasetConverters/ClassTypeGeneric.h"

bool replaceme(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

COCODatasetReader::COCODatasetReader(const std::string &path,const std::string& classNamesFile) {
    this->classNamesFile=classNamesFile;
    appendDataset(path);
}

COCODatasetReader::COCODatasetReader() {

}

bool COCODatasetReader::appendDataset(const std::string &datasetPath, const std::string &datasetPrefix) {
    std::cout << datasetPath << '\n';                //path to json Annotations file
    std::ifstream inFile(datasetPath);

    std::stringstream ss;

    if (inFile) {
      ss << inFile.rdbuf();
      inFile.close();
    }
    else {
      throw std::runtime_error("!! Unable to open json file");
    }

    rapidjson::Document doc;

    if (doc.Parse<0>(ss.str().c_str()).HasParseError())
        throw std::invalid_argument("json parse error");

    if(!(doc.HasMember("annotations") && doc.HasMember("images")))
        throw std::invalid_argument("Invalid Annotations file Passed");

    const rapidjson::Value& a = doc["annotations"];

    const rapidjson::Value& imgs = doc["images"];
    std::string filename(imgs[0]["file_name"].GetString(), imgs[0]["file_name"].GetStringLength());


    if(!a.IsArray())
        throw std::invalid_argument("Invalid Annotations file Passed");


    std::size_t path_last = datasetPath.find_last_of("/");

    std::size_t filename_first = filename.find_first_of("_");
    std::size_t filename_last = filename.find_last_of("_");

    std::string image_prefix_path = datasetPath.substr(0,path_last);
    std::string image_dir = filename.substr(filename_first + 1, filename_last - filename_first -1 );

    replaceme(image_prefix_path, "annotations", image_dir);

    std::string full_image_path = image_prefix_path + "/" + filename;


    for (rapidjson::Value::ConstValueIterator itr = a.Begin(); itr != a.End(); ++itr) {

        unsigned long int image_id = (*itr)["image_id"].GetUint64();
        //std::cout << image_id << '\n';
        int category = (*itr)["category_id"].GetUint();
        //std::cout << category << '\n';
        double x, y, w, h;
        x = (*itr)["bbox"][0].GetDouble();
        y = (*itr)["bbox"][1].GetDouble();
        w = (*itr)["bbox"][2].GetDouble();
        h = (*itr)["bbox"][3].GetDouble();


        if ( this->map_image_id.find(image_id) == this->map_image_id.end() ) {

            std::string num_string = std::to_string(image_id);

            std::size_t filename_id_start = full_image_path.find_last_of("_");
            std::size_t filename_ext = full_image_path.find_last_of(".");

            std::string dest = std::string( filename_ext - filename_id_start - 1 - num_string.length(), '0').append( num_string );


            full_image_path.replace(filename_id_start + 1, filename_ext - filename_id_start - 1, dest);


            Sample sample;
            sample.setSampleID(dest);
            sample.setColorImage(full_image_path);

            LOG(INFO) << "Loading Instance for Sample: " + full_image_path;

            RectRegionsPtr rectRegions(new RectRegions());
            ClassTypeGeneric typeConverter(this->classNamesFile);

            typeConverter.setId(category - 1);   //since index starts from 0 and categories from 1



            cv::Rect bounding(x , y , w , h);

            rectRegions->add(bounding,typeConverter.getClassString());

            sample.setRectRegions(rectRegions);

            this->samples.push_back(sample);

            this->map_image_id[image_id] = this->samples.size() - 1;
        } else {
            //this->samples[this->map_image_id[(*itr)["image_id"].GetUint64()]]

            ClassTypeGeneric typeConverter(this->classNamesFile);

            typeConverter.setId(category - 1);   //since index starts from 0 and categories from 1



            cv::Rect bounding(x , y , w , h);

            RectRegionsPtr rectRegions_old = this->samples[this->map_image_id[image_id]].getRectRegions();

            rectRegions_old->add(bounding,typeConverter.getClassString());

            this->samples[this->map_image_id[image_id]].setRectRegions(rectRegions_old);

            LOG(INFO) << "Loading Instance for Sample: " + this->samples[this->map_image_id[image_id]].getColorImagePath();

        }
        //this->map_image_id[this->samples.size()] = (*);


    }

    printDatasetStats();
}
