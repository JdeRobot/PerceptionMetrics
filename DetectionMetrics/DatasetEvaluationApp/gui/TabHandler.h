//
// Created by frivas on 19/02/17.
//

#ifndef SAMPLERGENERATOR_TABHANDLER_H
#define SAMPLERGENERATOR_TABHANDLER_H


#include <boost/shared_ptr.hpp>
#include <Utils/Configuration.h>
#include <QtWidgets/QMainWindow>
#include <mainwindow.h>

class TabHandler {
public:
    // Constructor function.
    TabHandler();
    // Get the context provided index.
    std::string getStringContext(int index);
    // Get entire an entire vector of elements present in context.
    std::vector<std::string> getAllContexts();

private:
    // A vector of strings to store different elements like "viewer","converter",etc.
    std::vector<std::string> contexts;
    // Fill "contexts" with certain elements.
    void fillContexts();
};


typedef boost::shared_ptr<TabHandler> TabHandlerPtr;

#endif //SAMPLERGENERATOR_TABHANDLER_H
