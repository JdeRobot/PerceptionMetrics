//
// Created by frivas on 19/02/17.
//

#include <glog/logging.h>
#include <DatasetConverters/readers/GenericDatasetReader.h>
#include "TabHandler.h"
#include "ListViewConfig.h"

// Contructor function(will be called whenever this object is created).
TabHandler::TabHandler() {
    fillContexts();
}


// Add "viewer" && "converter" to "contexts".
void TabHandler::fillContexts() {
    this->contexts.push_back("viewer");
    this->contexts.push_back("converter");
}

// Get the context of the handler provided index.
std::string TabHandler::getStringContext(int index) {
    return this->contexts[index];
}

// To get all the elements present in the "contexts"
std::vector<std::string> TabHandler::getAllContexts() {
    return this->contexts;
}
