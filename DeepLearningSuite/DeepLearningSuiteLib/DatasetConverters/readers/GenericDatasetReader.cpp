//
// Created by frivas on 4/02/17.
//

#include <glog/logging.h>
#include "GenericDatasetReader.h"


GenericDatasetReader::GenericDatasetReader(const std::string &path, const std::string& classNamesFile, const std::string &readerImplementation) {
    configureAvailablesImplementations(this->availableImplementations);
    if (std::find(this->availableImplementations.begin(), this->availableImplementations.end(), readerImplementation) != this->availableImplementations.end()){
        imp = getImplementation(readerImplementation);
        switch (imp) {
            case IMAGENET:
                this->imagenetDatasetReaderPtr = ImageNetDatasetReaderPtr( new ImageNetDatasetReader(path,classNamesFile));
                break;
            case COCO:
                this->cocoDatasetReaderPtr = COCODatasetReaderPtr( new COCODatasetReader(path,classNamesFile));
                break;
            case YOLO:
                this->yoloDatasetReaderPtr = YoloDatasetReaderPtr( new YoloDatasetReader(path,classNamesFile));
                break;
            case SPINELLO:
                this->spinelloDatasetReaderPtr = SpinelloDatasetReaderPtr( new SpinelloDatasetReader(path,classNamesFile));
                break;
            case OWN:
                this->ownDatasetReaderPtr = OwnDatasetReaderPtr( new OwnDatasetReader(path,classNamesFile));
                break;
            case PRINCETON:
                this->princetonDatasetReaderPtr = PrincetonDatasetReaderPtr( new PrincetonDatasetReader(path,classNamesFile));
                break;
            default:
                LOG(WARNING)<<readerImplementation + " is not a valid reader implementation";
                break;
        }
    }
    else{
        LOG(WARNING)<<readerImplementation + " is not a valid reader implementation";
    }
}


GenericDatasetReader::GenericDatasetReader(const std::vector<std::string> &paths, const std::string& classNamesFile,
                                           const std::string &readerImplementation) {
    configureAvailablesImplementations(this->availableImplementations);
    if (std::find(this->availableImplementations.begin(), this->availableImplementations.end(), readerImplementation) != this->availableImplementations.end()){
        imp = getImplementation(readerImplementation);
        switch (imp) {
            case YOLO:
                this->yoloDatasetReaderPtr = YoloDatasetReaderPtr( new YoloDatasetReader());
                for (auto it =paths.begin(), end= paths.end(); it != end; ++it){
                    int idx = std::distance(paths.begin(),it);
                    std::stringstream ss;
                    ss << idx << "_";
                    this->yoloDatasetReaderPtr->appendDataset(*it,ss.str());
                }
                break;
            case SPINELLO:
                this->spinelloDatasetReaderPtr = SpinelloDatasetReaderPtr( new SpinelloDatasetReader());
                for (auto it =paths.begin(), end= paths.end(); it != end; ++it){
                    int idx = std::distance(paths.begin(),it);
                    std::stringstream ss;
                    ss << idx << "_";
                    this->spinelloDatasetReaderPtr->appendDataset(*it,ss.str());
                }
                break;
            case OWN:
                this->ownDatasetReaderPtr = OwnDatasetReaderPtr( new OwnDatasetReader());
                for (auto it =paths.begin(), end= paths.end(); it != end; ++it){
                    int idx = std::distance(paths.begin(),it);
                    std::stringstream ss;
                    ss << idx << "_";
                    this->ownDatasetReaderPtr->appendDataset(*it,ss.str());
                }
                break;
            default:
                LOG(WARNING)<< readerImplementation + " is not a valid reader implementation";
                break;
        }
    }
    else{
        LOG(WARNING) << readerImplementation + " is not a valid reader implementation";
    }


}


void GenericDatasetReader::configureAvailablesImplementations(std::vector<std::string>& data) {
    data.push_back("imagenet");
    data.push_back("coco");
    data.push_back("yolo");
    data.push_back("spinello");
    data.push_back("own");
    data.push_back("princeton");

}

READER_IMPLEMENTATIONS GenericDatasetReader::getImplementation(const std::string& readerImplementation) {
    if (readerImplementation == "imagenet"){
        return IMAGENET;
    }
    if (readerImplementation == "coco"){
        return COCO;
    }
    if (readerImplementation == "yolo"){
        return YOLO;
    }
    if (readerImplementation == "spinello"){
        return SPINELLO;
    }
    if (readerImplementation == "own"){
        return OWN;
    }
    if (readerImplementation == "princeton"){
        return PRINCETON;
    }
}

DatasetReaderPtr GenericDatasetReader::getReader() {
    switch (imp) {
        case IMAGENET:
            return this->imagenetDatasetReaderPtr;
        case COCO:
            return this->cocoDatasetReaderPtr;
        case YOLO:
            return this->yoloDatasetReaderPtr;
        case SPINELLO:
            return this->spinelloDatasetReaderPtr;
        case OWN:
            return this->ownDatasetReaderPtr;
        case PRINCETON:
            return this->princetonDatasetReaderPtr;
        default:
            LOG(WARNING)<<imp + " is not a valid reader implementation";
            break;
    }
}

std::vector<std::string> GenericDatasetReader::getAvailableImplementations() {
    std::vector<std::string> data;

    configureAvailablesImplementations(data);
    return data;
}
