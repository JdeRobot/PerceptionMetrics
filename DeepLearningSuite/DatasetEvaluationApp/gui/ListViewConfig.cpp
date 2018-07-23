//
// Created by frivas on 18/02/17.
//

#include <QtCore/QStringListModel>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/algorithm/string.hpp>
#include <glog/logging.h>
#include <iostream>
#include "ListViewConfig.h"

bool ListViewConfig::configureDatasetInput(QMainWindow* mainWindow, QListView *qlistView, const std::string &path,bool multipleSelection) {

    if (!boost::filesystem::exists(boost::filesystem::path(path))){
        LOG(WARNING)<< "path: " + path  + " does not exist";
        return false;
    }


    QStringListModel *model;
    model = new QStringListModel(mainWindow);
    QStringList List;




    std::vector<std::string> filesID;

    getPathContentDatasetInput(path,filesID);


    std::sort(filesID.begin(),filesID.end());

    for (auto it = filesID.begin(), end = filesID.end(); it != end; ++it){
        std::string::size_type i = it->find(path);

        if (i != std::string::npos)
            it->erase(i, path.length());

        List << QString::fromStdString(*it);
    }

    model->setStringList(List);

    qlistView->setModel(model);
    if (multipleSelection)
        qlistView->setSelectionMode(QAbstractItemView::ExtendedSelection);

    return true;
}

void ListViewConfig::getPathContentDatasetInput(const std::string &path, std::vector<std::string>& content) {

    /*QString Qpath(path.c_str());
    QDirIterator itr(Qpath);
    while (itr.hasNext()) {
        itr.next();

        std::vector<std::string> possibleContent;
        if (itr.fileInfo().isDir() && itr.fileName().toStdString() != "." && itr.fileName().toStdString() != ".."){
            //check if yolo (should contain a *.txt
            bool isOwnFormat=false;
            bool takeParent=false;
            int skip_count = 0;
            QDirIterator itr2(itr.filePath());
            while (itr2.hasNext()) {
                itr2.next();

                if (itr2.fileName().toLower().toStdString().find(".txt") != std::string::npos){
                    possibleContent.push_back(itr2.filePath().toStdString());
                    takeParent = true;
                }
                else if(itr2.fileName().toLower().toStdString().find(".json") != std::string::npos){
                    possibleContent.push_back(itr2.filePath().toStdString());
                    takeParent = true;
                }
                else if(itr2.fileName().toLower().toStdString().find(".xml") != std::string::npos){
                    takeParent = true;
                    //Only Take Parent and break this will prevent displaying multiple xml files
                    break;
                    //possibleContent.push_back(itr2->path().string());
                }
                else if(itr2.fileInfo().completeSuffix().toLower().toStdString() == "png"
                        || itr2.fileInfo().completeSuffix().toLower().toStdString() == "jpeg"
                        || itr2.fileInfo().completeSuffix().toLower().toStdString() == "jpg"
                        || itr2.fileInfo().completeSuffix().toLower().toStdString() == "ppm"
                        || itr2.fileInfo().completeSuffix().toLower().toStdString() == "pgm")  {
                    if (skip_count >= 15) {         // If a directory contains more than 15 images, then it is aimge direcory for a dataset
                        break;                      // and it won't be indexed
                    }
                    skip_count++;
                }
                else if ((itr2.fileName().toLower().toStdString().find(".png") != std::string::npos) || (itr2.fileName().toLower().toStdString().find(".json") != std::string::npos)){
                    isOwnFormat=true;
                    break;
                }

            }



            if (takeParent) {
                possibleContent.push_back(itr.filePath().toStdString());
            }

            if (possibleContent.size() != 0){
                for (auto it = possibleContent.begin(), end = possibleContent.end(); it != end; ++it){
                    content.push_back(*it);
                }
            }
            else if (isOwnFormat){
                content.push_back(itr.filePath().toStdString());
            }
            else if (skip_count < 15){
                getPathContentDatasetInput(itr.filePath().toStdString(),content);
            }
        }

    }*/



    boost::filesystem::directory_iterator end_itr;
    boost::filesystem::path boostPath(path);
    std::size_t path_last;

    for (boost::filesystem::directory_iterator itr(boostPath); itr!=end_itr; ++itr)
    {
        std::vector<std::string> possibleContent;
        if (boost::filesystem::is_directory(*itr)){
            //check if yolo (should contain a *.txt
            bool isOwnFormat=false;
            bool takeParent=true;
            boost::filesystem::path boostPath2(itr->path());
            for (boost::filesystem::directory_iterator itr2(boostPath2); itr2!=end_itr; ++itr2) {
                if (itr2->path().string().find(".txt") != std::string::npos){
                    possibleContent.push_back(itr2->path().string());
                }
                else if(itr2->path().string().find(".json") != std::string::npos){
                    possibleContent.push_back(itr2->path().string());
                }
                else if(itr2->path().string().find(".xml") != std::string::npos){
                    //Only Take Parent and break this will prevent displaying multiple xml files
                    break;
                    //possibleContent.push_back(itr2->path().string());
                }
                else if ((itr2->path().string().find("png") != std::string::npos) || (itr2->path().string().find("json") != std::string::npos)){
                    isOwnFormat=true;
                    takeParent=false;
                    break;
                }
                else {
                    takeParent=false;
                }
            }
            if (takeParent) {
                possibleContent.push_back(itr->path().string());
            }

            if (possibleContent.size() != 0){
                for (auto it = possibleContent.begin(), end = possibleContent.end(); it != end; ++it){
                    content.push_back(*it);
                }
            }
            else if (isOwnFormat){
                content.push_back(itr->path().string());
            }
            else{
                getPathContentDatasetInput(itr->path().string(),content);
            }
        }
    }
}

bool ListViewConfig::configureInputByFile(QMainWindow *mainWindow, QListView *qlistView, const std::string &path,bool multipleSelection) {
    if (!boost::filesystem::exists(boost::filesystem::path(path))){
        LOG(WARNING) << "path: " + path  + " does not exist";
        return false;
    }


    QStringListModel *model;
    model = new QStringListModel(mainWindow);
    QStringList List;




    std::vector<std::string> filesID;

    getPathContentOnlyFiles(path,filesID);


    std::sort(filesID.begin(),filesID.end());

    for (auto it = filesID.begin(), end = filesID.end(); it != end; ++it){
        std::string::size_type i = it->find(path);

        if (i != std::string::npos)
            it->erase(i, path.length());

        List << QString::fromStdString(*it);
    }

    model->setStringList(List);

    qlistView->setModel(model);
    if (multipleSelection)
        qlistView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    return true;}

void ListViewConfig::getPathContentOnlyFiles(const std::string &path, std::vector<std::string> &content) {
    boost::filesystem::directory_iterator end_itr;
    boost::filesystem::path boostPath(path);

    for (boost::filesystem::directory_iterator itr(boostPath); itr!=end_itr; ++itr)
    {
        if (boost::filesystem::is_directory(*itr)){
            getPathContentOnlyFiles(itr->path().string(),content);
        }
        else{
            content.push_back(itr->path().string());
        }
    }
}

bool
ListViewConfig::configureInputByData(QMainWindow *mainWindow, QListView *qlistView, const std::vector<std::string>& data,bool multipleSelection) {
    QStringListModel *model;
    model = new QStringListModel(mainWindow);
    QStringList List;

    for (auto it = data.begin(), end = data.end(); it != end; ++it){
        List << QString::fromStdString(*it);
    }

    model->setStringList(List);

    qlistView->setModel(model);
    if (multipleSelection)
        qlistView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    return true;
}
