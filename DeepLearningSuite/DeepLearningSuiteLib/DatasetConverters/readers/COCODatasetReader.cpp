#include <fstream>
#include <glog/logging.h>
#include <boost/filesystem/path.hpp>
#include "COCODatasetReader.h"
#include "DatasetConverters/ClassTypeGeneric.h"

using namespace boost::filesystem;

bool replaceme(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

COCODatasetReader::COCODatasetReader(const std::string &path,const std::string& classNamesFile, bool imagesRequired):DatasetReader(imagesRequired) {
    this->classNamesFile=classNamesFile;
    appendDataset(path);
}


bool COCODatasetReader::find_img_directory(const path & dir_path, path & path_found, std::string& img_dirname) {
    std::cout << dir_path.string() << '\n';



    directory_iterator end_itr;

    int count = 0;

    for ( directory_iterator itr( dir_path ); itr != end_itr; ++itr ) {

        if (is_directory(itr->path())) {
            if (itr->path().filename().string() == img_dirname) {
                path_found = itr->path();
                return true;
            } else {
                if (find_img_directory( itr->path(), path_found , img_dirname) )
                return true;
            }

        }

    }
    return false;

}

bool COCODatasetReader::appendDataset(const std::string &datasetPath, const std::string &datasetPrefix) {
    std::cout << "Dataset Path: " << datasetPath << '\n';                //path to json Annotations file
    std::ifstream inFile(datasetPath);

    path boostDatasetPath(datasetPath);

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
        throw std::invalid_argument(std::string("JSON Parse Error: ") +  rapidjson::GetParseError_En(doc.GetParseError()));

    if( !doc.HasMember("annotations") )
        throw std::invalid_argument("Invalid Annotations file Passed");

    const rapidjson::Value& a = doc["annotations"];

    if(!a.IsArray())
        throw std::invalid_argument("Invalid Annotations file Passed, Images member isn't an array");


    std::string img_filename, img_dirname;
    std::size_t filename_id_start, filename_ext;

    if (this->imagesRequired || doc.HasMember("images")) {

        path img_dir;

        std::string filename = boostDatasetPath.filename().string();
        size_t first = filename.find_last_of('_');
        size_t last = filename.find_last_of('.');
        img_dirname = filename.substr(first + 1, last - first - 1);


        if (find_img_directory(boostDatasetPath.parent_path().parent_path(), img_dir, img_dirname)) {
            std::cout << "Image Directory Found: " << img_dir.string() << '\n';
        } else {
            throw std::invalid_argument("Corresponding Image Directory, can't be located, please place it in the same Directory as annotations"
            "If you wish to continue without reading images");

        }

        if(!doc.HasMember("images"))
            throw std::invalid_argument("Images Member not available, invalid annotations file passed");


        const rapidjson::Value& imgs = doc["images"];

        if(!imgs.IsArray())
            throw std::invalid_argument("Invalid Annotations file Passed, Images member isn't an array");


        for (rapidjson::Value::ConstValueIterator itr = imgs.Begin(); itr != imgs.End(); ++itr) {

            unsigned long int id = (*itr)["id"].GetUint64();
            std::string filename = (*itr)["file_name"].GetString();
            //std::cout << image_id << '\n';
            int category = (*itr)["category_id"].GetUint();

            Sample imsample;
            imsample.setSampleID(std::to_string(id));
            imsample.setColorImage(img_dir.string() + "/" + filename);

            this->map_image_id[id] = imsample;
        }

    }


    int counter = 0;

    for (rapidjson::Value::ConstValueIterator itr = a.Begin(); itr != a.End(); ++itr) {

        unsigned long int image_id = (*itr)["image_id"].GetUint64();
        //std::cout << image_id << '\n';
        int category = (*itr)["category_id"].GetUint();

        double x, y, w, h;
        x = (*itr)["bbox"][0].GetDouble();
        y = (*itr)["bbox"][1].GetDouble();
        w = (*itr)["bbox"][2].GetDouble();
        h = (*itr)["bbox"][3].GetDouble();
        bool isCrowd = (*itr).HasMember("iscrowd") ? ( (*itr)["iscrowd"].GetInt() > 0 ? true : false) : false;
        /*if (isCrowd) {
            std::cout << "Found 1" << '\n';
            exit(0);
        }*/
        //std::cout << isCrowd << '\n';
        //std::cout << x << y << w <<  h << '\n';
        //counter++;
        //if (counter == 100) {
        //    break;
        //}

        if ( this->map_image_id.find(image_id) == this->map_image_id.end() ) {

            std::string num_string = std::to_string(image_id);

            std::string full_image_path;


            Sample sample;
            sample.setSampleID(num_string);

            LOG(INFO) << "Loading Instance for Sample: " + num_string;

            RectRegionsPtr rectRegions(new RectRegions());
            ClassTypeGeneric typeConverter(this->classNamesFile);

            typeConverter.setId(category - 1);   //since index starts from 0 and categories from 1


            //std::cout << category << '\n';
            cv::Rect_<double> bounding = cv::Rect_<double>(x , y , w , h);


            if ((*itr).HasMember("score")) {
                //std::cout << "Adding Score" << '\n';
                rectRegions->add(bounding,typeConverter.getClassString(),(*itr)["score"].GetDouble(), isCrowd);
            } else {
                rectRegions->add(bounding,typeConverter.getClassString(), isCrowd);
            }
            sample.setRectRegions(rectRegions);

            //this->samples.push_back(sample);

            this->map_image_id[image_id] = sample;
        } else {
            //this->samples[this->map_image_id[(*itr)["image_id"].GetUint64()]]

            ClassTypeGeneric typeConverter(this->classNamesFile);

            typeConverter.setId(category - 1);   //since index starts from 0 and categories from 1



            cv::Rect_<double> bounding(x , y , w , h);


            Sample& sample = this->map_image_id[image_id];
            RectRegionsPtr rectRegions_old = sample.getRectRegions();

            //std::cout << "Initial Size" << rectRegions_old->getRegions().size() << '\n';

            if ((*itr).HasMember("score")) {
                //std::cout << "Adding Score" << '\n';
                rectRegions_old->add(bounding,typeConverter.getClassString(),(*itr)["score"].GetDouble(), isCrowd);
            } else {
                rectRegions_old->add(bounding,typeConverter.getClassString(), isCrowd);
            }

            sample.setRectRegions(rectRegions_old);

            //std::cout << "Size After: " << this->map_image_id[image_id].getRectRegions()->getRegions().size() << '\n';

            LOG(INFO) << "Loading Instance for Sample: " + sample.getSampleID();

        }
        //this->map_image_id[this->samples.size()] = (*);


    }

    this->samples.reserve(this->samples.size() + this->map_image_id.size());

    std::transform (this->map_image_id.begin(), this->map_image_id.end(),back_inserter(this->samples), [] (std::pair<unsigned long int, Sample> const & pair)
																				{
																					return pair.second;
																				});

    //printDatasetStats();
}
