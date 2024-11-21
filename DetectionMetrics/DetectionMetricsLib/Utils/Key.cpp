//
// Created by frivas on 4/02/17.
//

#include <sstream>
#include "Key.h"
#include <glog/logging.h>

Key::Key(const std::string &key):key(key) {

}


bool Key::isVector() {
    return this->values.size()>1;
}

void Key::addValue(const std::string &value) {
    this->values.push_back(value);
}

std::string Key::getValue() {
    if (this->values.size()==1)
        return this->values[0];
    else {
        const std::string ErrorMsg="Key [" + this->key + "] is an array not value";
        LOG(WARNING)<<ErrorMsg;
        throw ErrorMsg;
    }
}

std::string Key::getKey() {
    return this->key;
}

std::string Key::getValue(int id) {
    if (this->values.size()> id)
    {
        return this->values[id];
    }
}

std::string Key::getValueOrLast(int id) {
    return this->values[  this->values.size()> id ? id : this->values.size() - 1 ];
    // if id overflows return the last element of the array
}

std::vector<std::string> Key::getValues() {
    return this->values;
}

Key::Key() {

}

int Key::getNValues() {
    return this->values.size();
}

int Key::getValueAsInt() {
    if (this->values.size() != 1) {
        LOG(ERROR)<<"Cannot extract int from array type. Key=" + this->key;
        exit(1);
    }
    else{
        int value;
        std::istringstream iss(this->values[0]);
        iss >> value;
        return value;
    }
}
