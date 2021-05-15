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
 *       Victor Arribas Raigadas <.varribas.urjc@gmail.com>
 */

#ifndef STDUTILS_HPP
#define STDUTILS_HPP

//// string wrapper for stdlib.h::getenv

#include <cstdlib>
#include <string>

inline
std::string getEnvironmentVariable(std::string var){
	char* _env = getenv(var.c_str());
	return std::string((_env)?_env:"");
}


//// Fallback std::split
/// source: http://stackoverflow.com/questions/5607589/right-way-to-split-an-stdstring-into-a-vectorstring

#include <vector>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

namespace std {
inline
vector<string> split(string str, string del){
	vector<string> vstrings;
	boost::split(vstrings, str, boost::is_any_of(del));
	return vstrings;
}
}//NS



//// Check if file exists
/// For Linux works for files and directories
/// source: http://www.cplusplus.com/forum/general/1796/
#include <fstream>

namespace std {
inline
bool fileexists(std::string filepath){
	ifstream ifile(filepath.c_str(), ios_base::in);
	return ifile.is_open();
}
}//NS

#endif // STDUTILS_HPP
