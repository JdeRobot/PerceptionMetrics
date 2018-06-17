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

namespace Config{


Properties::Properties(){
}

Properties::Properties(YAML::Node node){
    this->node = node;
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


}//NS
