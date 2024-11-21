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

#ifndef JDEROBOT_CONFIG_CLASS_H
#define JDEROBOT_CONFIG_CLASS_H

/**
 * @mainpage  Config
 *            JdeRobot Config library
 *
 * @author    Aitor Martinez Fernandez , .aitor.martinez.fernandez@gmail.com
 * @date      September 2017
 * @version    0.9.0 (alpha)
 */

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <config/stdutils.hpp>

namespace Config{

class Properties {
public:
	Properties();
	Properties(YAML::Node node);
	//~Properties();

	void showConfig();

	void printNode(YAML::Node node, int nesting_level);

	bool keyExists(std::string element);

	bool NodeExists(YAML::Node n, std::vector<std::string> names);

	YAML::Node getNode(std::string element);
	/**
	 * @brief returns as string the propery given
	 *
	 * @param route to element separated by dots (example: "carViz.Camera.proxy")
	 *
	 */
	std::string asString(std::string element);


	/**
	 * @brief returns as string the propery given
	 *
	 * @param route to element separated by dots (example: "carViz.Camera.proxy")
	 * @param default value
	 *
	 */
	std::string asStringWithDefault(std::string element, std::string dataDefault);

	/**
	 * @brief returns as float the propery given
	 *
	 * @param route to element separated by dots (example: "carViz.Camera.proxy")
	 *
	 */
	float asFloat(std::string element);

	/**
	 * @brief returns as float the propery given
	 *
	 * @param route to element separated by dots (example: "carViz.Camera.proxy")
	 * @param default value
	 *
	 */
	float asFloatWithDefault(std::string element, float dataDefault);

	/**
	 * @brief returns as integer the propery given
	 *
	 * @param route to element separated by dots (example: "carViz.Camera.proxy")
	 *
	 */
	int asInt(std::string element);

	/**
	 * @brief returns as integer the propery given
	 *
	 * @param route to element separated by dots (example: "carViz.Camera.proxy")
	 * @param default value
	 *
	 */
	int asIntWithDefault(std::string element, int dataDefault);

	/**
	 * @brief returns as double the propery given
	 *
	 * @param route to element separated by dots (example: "carViz.Camera.proxy")
	 *
	 */
	double asDouble(std::string element);

	/**
	 * @brief returns as double the propery given
	 *
	 * @param route to element separated by dots (example: "carViz.Camera.proxy")
	 * @param default value
	 *
	 */
	double asDoubleWithDefault(std::string element, double dataDefault);



	YAML::Node getNode();

	void SetProperty(std::string key,std::string value);

private:
	YAML::Node node;

	/**
	 * @brief makes recursively sear for element given in names
	 *
	 *
	 * @param yaml node in which search
	 * @param vector of elements names (route to element of last position of vector)
	 *
	 *
	 * @return yaml node of element
	 */
	YAML::Node searchNode(YAML::Node n, std::vector<std::string> names);

};


/**
 * @brief function to make printable config class
 */
inline
std::ostream& operator<< (std::ostream & out, Properties & data) {
    out << data.getNode();
    return out ;
}

}//NS

#endif // JDEROBOT_CONFIG_CLASS_H
