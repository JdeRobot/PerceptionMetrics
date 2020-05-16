#include <fstream>
#include <glog/logging.h>
#include "ImageNetDatasetReader.h"
#include "DatasetConverters/ClassTypeGeneric.h"

using namespace boost::filesystem;


bool ImageNetDatasetReader::find_img_directory( const path & ann_dir_path, path & path_found ) {
    if ( !exists( ann_dir_path ) ) {
        return false;
    }
    directory_iterator end_itr;


    path parent_folder1 = ann_dir_path.parent_path();
    path parent_folder2 = parent_folder1.parent_path();

    for ( directory_iterator itr( parent_folder2 ); itr != end_itr; ++itr ) {
        if ( is_directory(itr->status()) ) {

            LOG(INFO) << itr->path().string() << '\n';
            if (itr->path().string() == parent_folder1.string()) {
                LOG(WARNING) << "skipping" << itr->path().string() << '\n';
                continue;
            } else if (itr->path().filename() == ann_dir_path.filename() ) {
                if ( find_directory(itr->path(), ann_dir_path.filename().string(), path_found ) )  // find the deepest nested directory
                    return true;

                path_found = itr->path();
                return true;
            } else {
                if ( find_directory( itr->path(), ann_dir_path.filename().string(), path_found ) )
                    return true;
            }
        }
    }
    return false;
}

bool ImageNetDatasetReader::find_directory(const path & dir_path, const std::string & dir_name, path & path_found) {

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

ImageNetDatasetReader::ImageNetDatasetReader(const std::string &path,const std::string& classNamesFile, bool imagesRequired):DatasetReader(imagesRequired) {
    this->classNamesFile=classNamesFile;
    appendDataset(path);
}

ImageNetDatasetReader::ImageNetDatasetReader(const std::string& classNamesFile, const bool imagesRequired):DatasetReader(imagesRequired) {
    this->classNamesFile=classNamesFile;
}

bool ImageNetDatasetReader::appendDataset(const std::string &datasetPath, const std::string &datasetPrefix) {
    boost::filesystem::path boostDatasetPath(datasetPath);

    if (!boost::filesystem::is_directory(boostDatasetPath)) {
        throw std::invalid_argument("Invalid File received for Imagenet Parser");
    }


    path img_dir;

    if (imagesRequired) {
        if (find_img_directory(boostDatasetPath, img_dir)) {
            LOG(INFO) << img_dir.string() << '\n';
            LOG(INFO) << "Image Directory Found" << '\n';
        } else {
            LOG(WARNING) << "Corresponding Image Directory, can't be located, Skipping" << '\n';
        }

    }


    boost::filesystem::directory_iterator end_itr;
    for (boost::filesystem::directory_iterator itr(boostDatasetPath); itr!=end_itr; ++itr)
    {
        if (!boost::filesystem::is_directory(*itr)){
            LOG(INFO) << itr->path().string() << '\n';
            boost::property_tree::ptree tree;

            boost::property_tree::read_xml(itr->path().string(), tree);

            std::string m_folder = tree.get<std::string>("annotation.folder");
            std::string m_filename = tree.get<std::string>("annotation.filename");
            std::string m_width = tree.get<std::string>("annotation.size.width");
            std::string m_height = tree.get<std::string>("annotation.size.height");




            Sample sample;
            sample.setSampleID(m_filename);

            if (imagesRequired) {
                std::string imgPath = img_dir.string() + "/" + m_filename + ".JPEG";
                sample.setColorImage(imgPath);

            }


            RectRegionsPtr rectRegions(new RectRegions());

            BOOST_FOREACH(boost::property_tree::ptree::value_type &v, tree.get_child("annotation")) {
        // The data function is used to access the data stored in a node.
                if (v.first == "object") {
                    std::string object_name = v.second.get<std::string>("name");
                    int xmin = v.second.get<int>("bndbox.xmin");
                    int xmax = v.second.get<int>("bndbox.xmax");
                    int ymin = v.second.get<int>("bndbox.ymin");
                    int ymax = v.second.get<int>("bndbox.ymax");

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
