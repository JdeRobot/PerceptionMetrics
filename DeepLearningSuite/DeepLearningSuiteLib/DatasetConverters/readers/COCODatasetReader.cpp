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

int LongestCommonSubsequence(std::string X, std::string Y)
{
    int m = X.length();
    int n = Y.length();

    int LCSuff[m+1][n+1];
    int result = 0;
    for (int i=0; i<=m; i++)
    {
        for (int j=0; j<=n; j++)
        {
            if (i == 0 || j == 0)
                LCSuff[i][j] = 0;

            else if (X[i-1] == Y[j-1])
            {
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1;
                result = std::max(result, LCSuff[i][j]);
            }
            else LCSuff[i][j] = 0;
        }
    }
    return result;
}

COCODatasetReader::COCODatasetReader(const std::string &path,const std::string& classNamesFile) {
    this->classNamesFile=classNamesFile;
    appendDataset(path);
}

COCODatasetReader::COCODatasetReader() {

}

bool COCODatasetReader::find_img_directory(const path & dir_path, path & path_found, std::string& img_filename, int& longestSubSeq) {

    directory_iterator end_itr;
    //int longestSubSeq = 0;


    for ( directory_iterator itr( dir_path ); itr != end_itr; ++itr ) {

        if (itr->path().has_extension() ) {

            std::string current_path = itr->path().extension().string();
            std::transform(current_path.begin(), current_path.end(), current_path.begin(), ::tolower);

            if ((current_path == ".jpg" || current_path == ".jpeg")) {
                int length = LongestCommonSubsequence(img_filename, itr->path().string());
                if (length > longestSubSeq) {
                    path_found = itr->path().parent_path();
                    longestSubSeq = length;

                }
                break;
            }

        }
        if ( is_directory(itr->path()) && find_img_directory( itr->path(), path_found , img_filename, longestSubSeq) )
                continue;
    }


    if (longestSubSeq >= 11 ) {
        return true;
    } else {
        return false;

    }
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

    bool read_images = true;

    if (!doc.HasMember("images")) {
        LOG(WARNING) << "Images Member not available, therefore images won't be read";
        read_images = false;
    }

    const rapidjson::Value& a = doc["annotations"];

    std::string filename, img_file_prefix;
    int prefix_length;

    if (read_images) {
        const rapidjson::Value& imgs = doc["images"];
        filename = std::string(imgs[0]["file_name"].GetString(), imgs[0]["file_name"].GetStringLength());
        prefix_length = filename.find_last_of('_');
        img_file_prefix = filename.substr(0, prefix_length);
    }

    if(!a.IsArray())
        throw std::invalid_argument("Invalid Annotations file Passed");

    path img_dir;

    if (read_images) {
        int longestSubSeq = 0;
        if (find_img_directory(boostDatasetPath.parent_path().parent_path(), img_dir, img_file_prefix, longestSubSeq)) {
            std::cout << "Image Directory Found: " << img_dir.string() << '\n';
        } else {
            throw std::runtime_error("Corresponding Image Directory, can't be located, please place it in the same Directory as annotations");
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

            if (read_images) {
                std::size_t filename_id_start = filename.find_last_of("_");
                std::size_t filename_ext = filename.find_last_of(".");

                std::string dest = std::string( filename_ext - filename_id_start - 1 - num_string.length(), '0').append( num_string );

                filename.replace(filename_id_start + 1, filename_ext - filename_id_start - 1, dest);

                full_image_path = img_dir.string() + "/" + filename;

            }

            Sample sample;
            sample.setSampleID(num_string);
            if (read_images)
                sample.setColorImage(full_image_path);

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

            this->samples.push_back(sample);

            this->map_image_id[image_id] = this->samples.size() - 1;
        } else {
            //this->samples[this->map_image_id[(*itr)["image_id"].GetUint64()]]

            ClassTypeGeneric typeConverter(this->classNamesFile);

            typeConverter.setId(category - 1);   //since index starts from 0 and categories from 1



            cv::Rect_<double> bounding(x , y , w , h);

            RectRegionsPtr rectRegions_old = this->samples[this->map_image_id[image_id]].getRectRegions();

            if ((*itr).HasMember("score")) {
                //std::cout << "Adding Score" << '\n';
                rectRegions_old->add(bounding,typeConverter.getClassString(),(*itr)["score"].GetDouble(), isCrowd);
            } else {
                rectRegions_old->add(bounding,typeConverter.getClassString(), isCrowd);
            }

            this->samples[this->map_image_id[image_id]].setRectRegions(rectRegions_old);

            LOG(INFO) << "Loading Instance for Sample: " + this->samples[this->map_image_id[image_id]].getSampleID();

        }
        //this->map_image_id[this->samples.size()] = (*);


    }

    //printDatasetStats();
}
