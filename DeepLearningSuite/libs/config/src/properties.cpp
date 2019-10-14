/*
 *  Copyright (C) 1997-2017 JDE Developers Team
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
 *       Aitor Martinez Fernandez <aitor.martinez.fernandez@gmail.com>
 */

#include <config/properties.hpp>
#include <glog/logging.h>
namespace Config{


Properties::Properties(){
}

Properties::Properties(YAML::Node node){
    this->node = node;
}

void
Properties::showConfig() {
    LOG(INFO) << "------------------------------------------------------------------" << std::endl;
    LOG(INFO) << "------------------------------------------------------------------" << std::endl;

    for (YAML::const_iterator it = this->node.begin(); it != this->node.end(); ++it){
       LOG(INFO) << it->first.as<std::string>() << ": ";
       printNode(it->second, 0);
       LOG(INFO) << '\n';
        // it->second.as<std::string>(); // can't do this until it's type is checked!!
   }

   LOG(INFO) << "------------------------------------------------------------------" << std::endl;
   LOG(INFO) << "------------------------------------------------------------------" << std::endl;

}

void
Properties::printNode(YAML::Node node_passed, int nesting_level) {
    //LOG(INFO) << nesting_level << '\n';
   switch (node_passed.Type()) {
     case  YAML::NodeType::Null:
        return;
     case YAML::NodeType::Scalar:
        LOG(INFO) << node_passed.as<std::string>() << '\n';
        break;
     case YAML::NodeType::Sequence:
        LOG(INFO) << '\n';
        for (YAML::const_iterator it = node_passed.begin(); it != node_passed.end(); ++it){
            LOG(INFO) << std::string(nesting_level, ' ')  << "-" << '\n';
            printNode(*it, nesting_level + 2);

            // it->second.as<std::string>(); // can't do this until it's type is checked!!
        }
        break;
     case YAML::NodeType::Map:
        for (YAML::const_iterator it = node_passed.begin(); it != node_passed.end(); ++it){
            LOG(INFO) << std::string(nesting_level, ' ') << it->first.as<std::string>() << ": ";
            printNode(it->second, nesting_level + 2);
         // it->second.as<std::string>(); // can't do this until it's type is checked!!
        }
        break;
     case YAML::NodeType::Undefined: // ...
        return;
   }
}

bool
Properties::keyExists(std::string element) {
    std::vector<std::string> v = std::split(element, ".");

    return this->NodeExists(this->node, v);

}

bool
Properties::NodeExists(YAML::Node n, std::vector<std::string> names) {
    YAML::Node nod = n[names[0]];
    names.erase(names.begin());

    if (names.size() > 0) {
        if (nod.IsSequence()) {
            for (YAML::const_iterator it=nod.begin();it!=nod.end();++it) {
                if (!this->NodeExists(*it, names))
                    return false;
            }
        } else {

              return this->searchNode(nod, names);
        }

    } else {
        return nod ? true : false;

    }


}

YAML::Node
Properties::getNode(std::string element) {
    std::vector<std::string> v = std::split(element, ".");

    YAML::Node nod = this->searchNode(this->node, v);
    return nod;
}

std::string
Properties::asString(std::string element){
    std::vector<std::string> v = std::split(element, ".");

    YAML::Node nod = this->searchNode(this->node, v);
    return nod.as<std::string>();
}

std::string
Properties::asStringWithDefault(std::string element, std::string dataDefault){
    std::vector<std::string> v = std::split(element, ".");

    YAML::Node nod = this->searchNode(this->node, v);
    std::string data;
    try{
        data = nod.as<std::string>();
    }catch(YAML::BadConversion e){
        data = dataDefault;
    }
    return data;
}

float
Properties::asFloat(std::string element){
    std::vector<std::string> v = std::split(element, ".");

    YAML::Node nod = this->searchNode(this->node, v);
    return nod.as<float>();
}

float
Properties::asFloatWithDefault(std::string element, float dataDefault){
    std::vector<std::string> v = std::split(element, ".");

    YAML::Node nod = this->searchNode(this->node, v);
    float data;
    try{
        data = nod.as<float>();
    }catch(YAML::BadConversion e){
        data = dataDefault;
    }
    return data;
}

int
Properties::asInt(std::string element){
    std::vector<std::string> v = std::split(element, ".");

    YAML::Node nod = this->searchNode(this->node, v);
    return nod.as<int>();
}

int
Properties::asIntWithDefault(std::string element, int dataDefault){
    std::vector<std::string> v = std::split(element, ".");

    YAML::Node nod = this->searchNode(this->node, v);
    int data;
    try{
        data = nod.as<int>();
    }catch(YAML::BadConversion e){
        data = dataDefault;
    }
    return data;
}

double
Properties::asDouble(std::string element){
    std::vector<std::string> v = std::split(element, ".");

    YAML::Node nod = this->searchNode(this->node, v);
    return nod.as<double>();
}

double
Properties::asDoubleWithDefault(std::string element, double dataDefault){
    std::vector<std::string> v = std::split(element, ".");

    YAML::Node nod = this->searchNode(this->node, v);
    double data;
    try{
        data = nod.as<double>();
    }catch(YAML::BadConversion e){
        data = dataDefault;
    }
    return data;
}

YAML::Node
Properties::getNode(){

    return node;
}



YAML::Node
Properties::searchNode(YAML::Node n, std::vector<std::string> names){
    YAML::Node nod = n[names[0]];
    names.erase(names.begin());

    if (names.size()>0){
        return this->searchNode(nod, names);
    }else{
        return nod;
    }
}

void Properties::SetProperty(std::string key, std::string value){
  this->node[key] = value;
}

}//NS
