//
// Created by frivas on 4/02/17.
//

#include <glog/logging.h>
#include "GenericDatasetReader.h"


GenericDatasetReader::GenericDatasetReader(const std::string &path, const std::string& classNamesFile, const std::string &readerImplementation, bool imagesRequired) {
    configureAvailablesImplementations(this->availableImplementations);
    if (std::find(this->availableImplementations.begin(), this->availableImplementations.end(), readerImplementation) != this->availableImplementations.end()){
        imp = getImplementation(readerImplementation);
        switch (imp) {
	    case OPENIMAGES:
                this->openimagesDatasetReaderPtr = OpenImagesDatasetReaderPtr( new OpenImagesDatasetReader(path,classNamesFile,imagesRequired));
                break;
	    case IMAGENET:
                this->imagenetDatasetReaderPtr = ImageNetDatasetReaderPtr( new ImageNetDatasetReader(path,classNamesFile,imagesRequired));
                break;
            case COCO:
                this->cocoDatasetReaderPtr = COCODatasetReaderPtr( new COCODatasetReader(path,classNamesFile,imagesRequired));
                break;
            case PASCALVOC:
                this->pascalvocDatasetReaderPtr = PascalVOCDatasetReaderPtr( new PascalVOCDatasetReader(path,classNamesFile,imagesRequired));
                break;
            case YOLO_1:
                this->yoloDatasetReaderPtr = YoloDatasetReaderPtr( new YoloDatasetReader(path,classNamesFile,imagesRequired));
                break;
            case SPINELLO:
                this->spinelloDatasetReaderPtr = SpinelloDatasetReaderPtr( new SpinelloDatasetReader(path,classNamesFile,imagesRequired));
                break;
            case OWN:
                this->ownDatasetReaderPtr = OwnDatasetReaderPtr( new OwnDatasetReader(path,classNamesFile,imagesRequired));
                break;
            case PRINCETON:
                this->princetonDatasetReaderPtr = PrincetonDatasetReaderPtr( new PrincetonDatasetReader(path,classNamesFile,imagesRequired));
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
                                           const std::string &readerImplementation, bool imagesRequired) {
    configureAvailablesImplementations(this->availableImplementations);
    if (std::find(this->availableImplementations.begin(), this->availableImplementations.end(), readerImplementation) != this->availableImplementations.end()){
        imp = getImplementation(readerImplementation);
        switch (imp) {
            case IMAGENET:
                this->imagenetDatasetReaderPtr = ImageNetDatasetReaderPtr( new ImageNetDatasetReader(classNamesFile, imagesRequired));
                for (auto it =paths.begin(), end= paths.end(); it != end; ++it){
                    int idx = std::distance(paths.begin(),it);
                    std::stringstream ss;
                    ss << idx << "_";
                    this->imagenetDatasetReaderPtr->appendDataset(*it,ss.str());
                }
                break;
            case YOLO_1:
                this->yoloDatasetReaderPtr = YoloDatasetReaderPtr( new YoloDatasetReader(classNamesFile, imagesRequired));
                for (auto it =paths.begin(), end= paths.end(); it != end; ++it){
                    int idx = std::distance(paths.begin(),it);
                    std::stringstream ss;
                    ss << idx << "_";
                    this->yoloDatasetReaderPtr->appendDataset(*it,ss.str());
                }
                break;
            case SPINELLO:
                this->spinelloDatasetReaderPtr = SpinelloDatasetReaderPtr( new SpinelloDatasetReader(imagesRequired));
                for (auto it =paths.begin(), end= paths.end(); it != end; ++it){
                    int idx = std::distance(paths.begin(),it);
                    std::stringstream ss;
                    ss << idx << "_";
                    this->spinelloDatasetReaderPtr->appendDataset(*it,ss.str());
                }
                break;
            case OWN:
                this->ownDatasetReaderPtr = OwnDatasetReaderPtr( new OwnDatasetReader(imagesRequired));
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
    data.push_back("Open Images");
    data.push_back("ImageNet");
    data.push_back("COCO");
    data.push_back("Pascal VOC");
    data.push_back("YOLO");
    data.push_back("Spinello");
    data.push_back("Own");
    data.push_back("Princeton");

}

READER_IMPLEMENTATIONS GenericDatasetReader::getImplementation(const std::string& readerImplementation) {
    if (readerImplementation == "Open Images") {
	return OPENIMAGES;
    }
    if (readerImplementation == "ImageNet"){
        return IMAGENET;
    }
    if (readerImplementation == "COCO"){
        return COCO;
    }
    if (readerImplementation == "Pascal VOC"){
        return PASCALVOC;
    }
    if (readerImplementation == "YOLO"){
        return YOLO_1;
    }
    if (readerImplementation == "Spinello"){
        return SPINELLO;
    }
    if (readerImplementation == "Own"){
        return OWN;
    }
    if (readerImplementation == "Princeton"){
        return PRINCETON;
    }
}

DatasetReaderPtr GenericDatasetReader::getReader() {
    switch (imp) {
	case OPENIMAGES:
            return this->openimagesDatasetReaderPtr;
        case IMAGENET:
            return this->imagenetDatasetReaderPtr;
        case COCO:
            return this->cocoDatasetReaderPtr;
        case PASCALVOC:
            return this->pascalvocDatasetReaderPtr;
        case YOLO_1:
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
