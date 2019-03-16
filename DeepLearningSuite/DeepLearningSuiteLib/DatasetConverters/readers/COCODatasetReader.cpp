#include <fstream>
#include <glog/logging.h>
#include <boost/filesystem/path.hpp>
#include "COCODatasetReader.h"

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
    LOG(INFO) << dir_path.string() << '\n';



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
    LOG(INFO) << "Dataset Path: " << datasetPath << '\n';                //path to json Annotations file
    std::ifstream inFile(datasetPath);

    path boostDatasetPath(datasetPath);

    ClassTypeGeneric typeConverter(this->classNamesFile);

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
            LOG(INFO) << "Image Directory Found: " << img_dir.string() << '\n';
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

            int category = (*itr)["category_id"].GetUint();

            Sample imsample;
            imsample.setSampleID(std::to_string(id));
            imsample.setColorImage(img_dir.string() + "/" + filename);
            if ( itr->HasMember("width") && itr->HasMember("height") ) {
                imsample.setSampleDims((*itr)["width"].GetInt(), (*itr)["height"].GetInt());
            }

            this->map_image_id[id] = imsample;
        }

    }


    int counter = 0;

    bool hasBbox = true;

    for (rapidjson::Value::ConstValueIterator itr = a.Begin(); itr != a.End(); ++itr) {

        unsigned long int image_id = (*itr)["image_id"].GetUint64();

        int category = (*itr)["category_id"].GetUint();

        hasBbox = (*itr).HasMember("bbox");
        double x, y, w, h;

        if (hasBbox) {
            x = (*itr)["bbox"][0].GetDouble();
            y = (*itr)["bbox"][1].GetDouble();
            w = (*itr)["bbox"][2].GetDouble();
            h = (*itr)["bbox"][3].GetDouble();

        }
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


            typeConverter.setId(category - 1);   //since index starts from 0 and categories from 1

            if (hasBbox) {

                RectRegionsPtr rectRegions(new RectRegions());

                cv::Rect_<double> bounding = cv::Rect_<double>(x , y , w , h);


                if ((*itr).HasMember("score")) {
                    //Adding Score
                    rectRegions->add(bounding,typeConverter.getClassString(),(*itr)["score"].GetDouble(), isCrowd);
                } else {
                    rectRegions->add(bounding,typeConverter.getClassString(), isCrowd);
                }
                sample.setRectRegions(rectRegions);
            }


            if ((*itr).HasMember("segmentation"))
                appendSegmentationRegion(*itr, sample, typeConverter, isCrowd);

            //this->samples.push_back(sample);

            this->map_image_id[image_id] = sample;
        } else {
            //this->samples[this->map_image_id[(*itr)["image_id"].GetUint64()]]


            typeConverter.setId(category - 1);   //since index starts from 0 and categories from 1

            Sample& sample = this->map_image_id[image_id];

            if (hasBbox) {

                cv::Rect_<double> bounding(x , y , w , h);

                RectRegionsPtr rectRegions_old = sample.getRectRegions();

                if ((*itr).HasMember("score")) {
                    //Adding Score
                    rectRegions_old->add(bounding,typeConverter.getClassString(),(*itr)["score"].GetDouble(), isCrowd);
                } else {
                    rectRegions_old->add(bounding,typeConverter.getClassString(), isCrowd);
                }

                sample.setRectRegions(rectRegions_old);
            }



            if ((*itr).HasMember("segmentation"))
                appendSegmentationRegion(*itr, sample, typeConverter, isCrowd);


            LOG(INFO) << "Loading Instance for Sample: " + sample.getSampleID();

        }


    }

    this->samples.reserve(this->samples.size() + this->map_image_id.size());

    std::transform (this->map_image_id.begin(), this->map_image_id.end(),back_inserter(this->samples), [] (std::pair<unsigned long int, Sample> const & pair)
																				{
																					return pair.second;
																				});

    //printDatasetStats();
}


void COCODatasetReader::appendSegmentationRegion(const rapidjson::Value& node, Sample& sample, ClassTypeGeneric typeConverter, const bool isCrowd) {


    RLE region = getSegmentationRegion(node["segmentation"], sample.getSampleWidth(), sample.getSampleHeight());
    //std::cout << "RLE String: " << rleToString( &region ) << '\n';
    RleRegionsPtr rleRegions = sample.getRleRegions();
    std::string className = typeConverter.getClassString();
    if (node.HasMember("score")) {
        rleRegions->add(region, className, node["score"].GetDouble(), isCrowd);
    } else {
        rleRegions->add(region, className, isCrowd);
    }
    sample.setRleRegions(rleRegions);

}


RLE COCODatasetReader::getSegmentationRegion(const rapidjson::Value& seg, int im_width, int im_height) {

    if (seg.IsArray()) {

        if (!seg.Empty()) {

            if (seg[0].IsArray()) {       // Multiple Arrays

                return fromSegmentationList(seg, im_width, im_height, (int)seg.Size());


            } else if (seg[0].IsObject()) {                     // list of objects, size is available no need to store

                return fromSegmentationObject(seg, seg.Size());

            } else if (seg[0].IsDouble() || seg[0].IsInt()) {

                return fromSegmentationList(seg, im_width, im_height, 0);

            }

        }

    } else if (seg.IsObject()) {

        return fromSegmentationObject(seg, 0);

    } else {
        LOG(WARNING) << "Invalid segmentation Annotations, skipping";
    }



}

RLE COCODatasetReader::fromSegmentationObject(const rapidjson::Value& seg, int size) {

    if (size == 0) {                // single object
        if (seg.HasMember("counts")) {
            const rapidjson::Value& counts = seg["counts"];
            if (counts.IsArray()) {
                return fromUncompressedRle(seg);
            } else if (counts.IsString()) {
                return fromRle(seg);
            } else {
                throw std::invalid_argument("Invalid Annotations File Passed\n Segmentation Member has an invalid counts member");
            }
        }
    }


    RLE* multipleRles;
    rlesInit(&multipleRles, size);
    for ( int i = 0; i < size; i++) {

        if (seg[i].HasMember("counts")) {
            const rapidjson::Value& counts = seg[i]["counts"];
            if (counts.IsArray()) {
                multipleRles[i] = fromUncompressedRle(seg[i]);
            } else if (counts.IsString()) {
                multipleRles[i] = fromRle(seg[i]);
            } else {
                throw std::invalid_argument("Invalid Annotations File Passed\n Segmentation Member has an invalid counts member");
            }
        }

    }
    RLE* resultingRle;
    rlesInit(&resultingRle, 1);
    rleMerge(multipleRles, resultingRle, size, 0);

    rlesFree(&multipleRles, size);


    /*cv::Mat matrix_decoded(resultingRle->h, resultingRle->w, CV_8U);

    rleDecode(resultingRle, matrix_decoded.data , 1);
    matrix_decoded = matrix_decoded * 255;

    cv::imshow("From Seg Object", matrix_decoded);
    cv::waitKey(0);
    */
    return *resultingRle;
}

RLE COCODatasetReader::fromUncompressedRle(const rapidjson::Value& seg) {

    RLE result;

    const rapidjson::Value& arr = seg["counts"];
    uint* data = (uint*) malloc((int)(arr.Size()* sizeof(uint)));
    unsigned long i;
    for (i = 0; i < arr.Size(); i++) {
        data[i] = (uint) arr[i].GetUint();
    }

    rleInit(&result, seg["size"][0].GetInt64(), seg["size"][1].GetInt64(), i, data );


    /*cv::Mat matrix_decoded(result.w, result.h, CV_8U);

    rleDecode(&result, matrix_decoded.data , 1);
    matrix_decoded = matrix_decoded * 255;

    //cv::bitwise_not(matrix_decoded, matrix_decoded);

    //cv::rotate(matrix_decoded, matrix_decoded, cv::ROTATE_90_CLOCKWISE);
    //cv::flip(matrix_decoded, matrix_decoded, 1);


    /*cv::imshow("From Uncompressed", matrix_decoded);
    cv::waitKey(0);

    std::cout << "In debug: " << rleToString(&result) << '\n';
    */
    return result;
    /*# time for malloc can be saved here but
     its fine
    data = <uint*> malloc(len(cnts)* sizeof(uint))
    for j in range(len(cnts)):
        data[j] = <uint> cnts[j]

    R = RLE(ucRles[i]['size'][0], ucRles[i]['size'][1], len(cnts), <uint*> data)
    Rs._R[0] = R
    objs.append(_toString(Rs)[0])*/

}

RLE COCODatasetReader::fromSegmentationList(const rapidjson::Value& seg, int im_width, int im_height, int size) {


    if (size == 0) {
        RLE result;
        double* arr = new double[seg.Size()];
        int i;
        for (i = 0; i < seg.Size(); i++) {
            arr[i] = seg[i].GetDouble();
        }
        rleFrPoly( &result, arr, i/2 , im_height, im_width);
        //cv::Mat matrix_decoded(result.h, result.w, CV_8U);

        /*rleDecode(&result, matrix_decoded.data , 1);
        matrix_decoded = matrix_decoded * 255;

        cv::imshow("From List 0", matrix_decoded);
        cv::waitKey(0);*/
        return result;

    } else {

        RLE* multipleRles;
        rlesInit(&multipleRles, size);
        for (int i = 0; i < size; i++) {
            if (seg[i].IsArray()) {
                double* arr = new double[(int)(seg[i].Size())];
                int j;
                for (j = 0; j < (int)seg[i].Size(); j++) {
                    arr[j] = seg[i][j].GetDouble();
                }
                rleFrPoly( multipleRles + i, arr, j/2 , im_height, im_width);
            } else {
                throw std::invalid_argument("Invalid Annotations File Passed\n Error Detected in segmentation Member, 2D array consists of a Scalar");
            }


        }
        RLE* resultingRle;
        rlesInit(&resultingRle, 1);
        rleMerge(multipleRles, resultingRle, size, 0);

        //rlesFree(&multipleRles, size);

        /*std::cout << "In debug: " << rleToString(resultingRle) << '\n';

        //std::cout << result1.h << " " << result1.w << '\n';

        cv::Mat matrix_decoded(resultingRle->w, resultingRle->h, CV_8U);

        rleDecode(resultingRle, matrix_decoded.data , 1);
        matrix_decoded = matrix_decoded * 255;

        cv::imshow("From List", matrix_decoded);
        cv::waitKey(0);
        */

        return *resultingRle;

    }

}

RLE COCODatasetReader::fromRle(const rapidjson::Value& seg) {

    RLE result;
    rleFrString( &result, (char*) seg["counts"].GetString(), seg["size"][0].GetUint() , seg["size"][1].GetUint() );


    /*cv::Mat matrix_decoded(result.h, result.w, CV_8U);

    rleDecode(&result, matrix_decoded.data , 1);
    matrix_decoded = matrix_decoded * 255;

    cv::rotate(matrix_decoded, matrix_decoded, cv::ROTATE_90_CLOCKWISE);
    cv::flip(matrix_decoded, matrix_decoded, 1);

    cv::imshow("From RLE", matrix_decoded);
    cv::waitKey(0);
    */


    return result;
}
