//
// Created by frivas on 18/02/17.
//

#ifndef SAMPLERGENERATOR_LISTVIEWCONFIG_H
#define SAMPLERGENERATOR_LISTVIEWCONFIG_H

#include <string>
#include <QtWidgets/QListView>
#include <QtWidgets/QMainWindow>


class ListViewConfig {
public:
    static bool configureDatasetInput(QMainWindow* mainWindow, QListView* qlistView, const std::string& path, bool multipleSelection);
    static bool configureInputByFile(QMainWindow* mainWindow, QListView* qlistView, const std::string& path, const std::string& pathIdentifier, bool multipleSelection);
    static bool configureInputByData(QMainWindow* mainWindow, QListView* qlistView, const std::vector<std::string>& data,bool multipleSelection);



private:
    static void getPathContentDatasetInput(const std::string& path, std::vector<std::string>& content);
    static void getPathContentOnlyFiles(const std::string& path, std::vector<std::string>& content);

};

#endif //SAMPLERGENERATOR_LISTVIEWCONFIG_H
