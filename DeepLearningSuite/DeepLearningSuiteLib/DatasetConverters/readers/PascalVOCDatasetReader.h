#ifndef SAMPLERGENERATOR_PASCALVOCDATASETREADER_H
#define SAMPLERGENERATOR_PASCALVOCDATASETREADER_H


#include <DatasetConverters/readers/DatasetReader.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/foreach.hpp>


class PascalVOCDatasetReader: public DatasetReader {
public:
    PascalVOCDatasetReader(const std::string& path,const std::string& classNamesFile, const bool imagesRequired);
    bool appendDataset(const std::string& datasetPath, const std::string& datasetPrefix="");
    bool find_directory(const boost::filesystem::path & dir_path, const std::string & dir_name, boost::filesystem::path & path_found);

};

typedef boost::shared_ptr<PascalVOCDatasetReader> PascalVOCDatasetReaderPtr;

#endif //SAMPLERGENERATOR_PascalVOCDATASETREADER_H
