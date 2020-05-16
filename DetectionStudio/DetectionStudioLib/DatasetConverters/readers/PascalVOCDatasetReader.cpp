#include <fstream>
#include <glog/logging.h>
#include <boost/filesystem/path.hpp>
#include "PascalVOCDatasetReader.h"
#include "DatasetConverters/ClassTypeGeneric.h"

using namespace boost::filesystem;

PascalVOCDatasetReader::PascalVOCDatasetReader(const std::string &path,const std::string& classNamesFile, const bool imagesRequired):DatasetReader(imagesRequired) {
    this->classNamesFile=classNamesFile;
    appendDataset(path);
}


bool PascalVOCDatasetReader::find_directory(const path & dir_path, const std::string & dir_name, path & path_found) {

    directory_iterator end_itr;


    for ( directory_iterator itr( dir_path ); itr != end_itr; ++itr ) {
        if ( is_directory(itr->status()) ) {

            if (itr->path().filename() == dir_name ) {
                if ( find_directory(itr->path(), dir_name, path_found ) )  // find the deepest nested directory
                    return true;

                path_found = itr->path();
                return true;
            } else {
                if ( find_directory( itr->path(), dir_name, path_found ) )
                    return true;
            }
        }
    }
    return false;
}


bool PascalVOCDatasetReader::appendDataset(const std::string &datasetPath, const std::string &datasetPrefix) {

    boost::filesystem::path boostDatasetPath(datasetPath);

    if (boost::filesystem::exists(boostDatasetPath)) {
        if (!boost::filesystem::is_directory(boostDatasetPath)) {
            throw std::invalid_argument("Please Provide a folder containing all the annotation files, not just a single file");
        }
    } else {
        throw std::invalid_argument("Provided Directory Path doesn't exist");
    }

    path img_dir;


    if (imagesRequired) {
        if (find_directory(boostDatasetPath.parent_path(), "JPEGImages", img_dir)) {
            LOG(INFO) << img_dir.string() << '\n';
        } else {
            throw std::runtime_error("Images Directory can't be located, place it in the folder containing annotations, and name it JPEGIamges");
        }
    }


    int count = 0;

    boost::filesystem::directory_iterator end_itr;
    for (boost::filesystem::directory_iterator itr(boostDatasetPath); itr!=end_itr; ++itr)
    {
        if (!boost::filesystem::is_directory(*itr)){
            count++;


            LOG(INFO) << itr->path().string() << '\n';
            boost::property_tree::ptree tree;

            boost::property_tree::read_xml(itr->path().string(), tree);


            std::string m_id  = itr->path().stem().string();    // filename without extension
            std::string m_imgfile = tree.get<std::string>("annotation.filename");
            std::string m_width = tree.get<std::string>("annotation.size.width");
            std::string m_height = tree.get<std::string>("annotation.size.height");



            Sample sample;
            sample.setSampleID(m_id);

            if (imagesRequired) {
                std::string imgPath = img_dir.string() + "/" + m_imgfile;
                sample.setColorImage(imgPath);

            }


            RectRegionsPtr rectRegions(new RectRegions());

            BOOST_FOREACH(boost::property_tree::ptree::value_type &v, tree.get_child("annotation")) {
        // The data function is used to access the data stored in a node.
                if (v.first == "object") {
                    std::string object_name = v.second.get<std::string>("name");
                    int xmin = int(v.second.get<double>("bndbox.xmin"));
                    int xmax = int(v.second.get<double>("bndbox.xmax"));
                    int ymin = int(v.second.get<double>("bndbox.ymin"));
                    int ymax = int(v.second.get<double>("bndbox.ymax"));

                    cv::Rect bounding(xmin, ymin, xmax - xmin, ymax - ymin);
                    rectRegions->add(bounding,object_name);

                }
            }

            sample.setRectRegions(rectRegions);
            this->samples.push_back(sample);

        }
    }



    printDatasetStats();
}
