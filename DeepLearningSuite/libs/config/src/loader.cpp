/*
 *  Copyright (C) 1997-2015 JDE Developers Team
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see http://www.gnu.org/licenses/.
 *  Authors :
 *       Victor Arribas Raigadas <.varribas.urjc@gmail.com>
 */


#include <config/loader.hpp>
#include <glog/logging.h>
namespace jderobotconfig{
namespace loader{

std::string
findConfigFile(const std::string& filename){
    if (std::fileexists(filename))
        return filename;

    std::string path_holders[] = {getEnvironmentVariable(CONFIG_PATH_NAME)};

    for (int i=0; i<2; i++){
        if (path_holders[i].empty()) continue;
        for (std::string path : std::split(path_holders[i], ":")){
            if (path.empty()) continue;
            std::string filepath(path+"/"+filename);
            if (std::fileexists(filepath))
                return filepath;
        }
    }

    return "";
}

Config::Properties
load(std::string filename){
    std::string filepath = findConfigFile(filename);
    if (filepath.empty()){
        YAML::Exception e(YAML::Mark(),"jderobot/config/loader.cpp: file " + filepath + " Not Found");
        throw e;
    }
    YAML::Node nodeConfig = YAML::LoadFile(filepath);

    Config::Properties config(nodeConfig);
    LOG(INFO)<<"[Info] loaded YAML Config file: "<<filepath<<std::endl;
    //properties->setProperty("Ice.Config", filepath);
    return config;
}


}}//NS
