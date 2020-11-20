#include <fstream>
#include <glog/logging.h>
#include <boost/filesystem/path.hpp>
#include "OpenImagesDatasetReader.h"
#include "DatasetConverters/ClassTypeGeneric.h"

#include <istream>
#include <string>
#include <vector>


using namespace boost::filesystem;



OpenImagesDatasetReader::OpenImagesDatasetReader(const std::string &path,const std::string& classNamesFile, bool imagesRequired):DatasetReader(imagesRequired) {
    this->classNamesFile=classNamesFile;
    appendDataset(path);
}

bool OpenImagesDatasetReader::find_img_directory(const path & dir_path, path & path_found) {
    LOG(INFO) << dir_path.string() << '\n';
    /*directory_iterator end_itr;
    int count = 0;

    for (directory_iterator itr(dir_path); itr != end_itr; ++itr) {
        if (is_directory(itr->path())) {
            if (itr->path().filename().string() == img_dirname) {
                path_found = itr->path();
                return true;
            } else {
                if (find_img_directory(itr->path(), path_found , img_dirname))
                return true;
           }
        }

    }*/
    return false;
}


enum class CSVState {
    UnquotedField,
    QuotedField,
    QuotedQuote
};

std::vector<std::string> readCSVRow(const std::string &row) {
	//LOG(WARNING) << row << "\n";
    CSVState state = CSVState::UnquotedField;
    std::vector<std::string> fields {""};
    size_t i = 0; // index of the current field
    for (char c : row) {
        switch (state) {
            case CSVState::UnquotedField:
                switch (c) {
                    case ',': // end of field
                              fields.push_back(""); i++;
                              break;
                    case '"': state = CSVState::QuotedField;
                              break;
                    default:  fields[i].push_back(c);
                              break; }
                break;
            case CSVState::QuotedField:
                switch (c) {
                    case '"': state = CSVState::QuotedQuote;
                              break;
                    default:  fields[i].push_back(c);
                              break; }
                break;
            case CSVState::QuotedQuote:
                switch (c) {
                    case ',': // , after closing quote
                              fields.push_back(""); i++;
                              state = CSVState::UnquotedField;
                              break;
                    case '"': // "" -> "
                              fields[i].push_back('"');
                              state = CSVState::QuotedField;
                              break;
                    default:  // end of quote
                              state = CSVState::UnquotedField;
                              break; }
                break;
        }
    }
    return fields;
}

/// Read CSV file, Excel dialect. Accept "quoted fields ""with quotes"""
std::vector<std::vector<std::string>> readCSV(std::istream &in) {
    std::vector<std::vector<std::string>> table;
    std::string row;
    int max_counter = 50;
    int counter = 0;
    while (!in.eof() and counter < max_counter) {
        std::getline(in, row);
        if (in.bad() || in.fail()) {
            break;
        }
        auto fields = readCSVRow(row);
	//for (auto i = fields.begin(); i != fields.end(); ++i)
    	//	std::cout << *i << ' ';
	LOG(WARNING) << fields[0] << "\n";
        table.push_back(fields);
	counter++;
    }
    return table;
}




bool OpenImagesDatasetReader::appendDataset(const std::string &datasetPath, const std::string &datasetPrefix) {
    LOG(INFO) << "Dataset Path: " << datasetPath << '\n';
	


	std::ifstream file(datasetPath);
	readCSV(file);    

/*	std::ifstream inFile(datasetPath);
    path boostDatasetPath(datasetPath);
    ClassTypeGeneric typeConverter(this->classNamesFile);
    std::stringstream ss;
LOG(INFO) << "1" << '\n';
    if (inFile) {
  LOG(INFO) << "1.1" << '\n';
	    ss << inFile.rdbuf();
      LOG(INFO) << "1.2" << '\n';
	    inFile.close();
      LOG(INFO) << "1.3" << '\n';
	    LOG(WARNING) << ss.str()<< "\n";
    } else {
      throw std::runtime_error("!! Unable to open json file");
    }
LOG(INFO) << "2" << '\n';*/
}

