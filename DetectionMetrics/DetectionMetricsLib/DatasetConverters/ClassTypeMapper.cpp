#include "ClassTypeMapper.h"

ClassTypeMapper::ClassTypeMapper(const std::string& classNamesFile) {

    this->root = Tree("../ClassMappingHierarchy.xml");             // Initializing tree for mapping synonmys
    fillStringClassesVector(classNamesFile);

}

ClassTypeMapper::ClassTypeMapper() {

}

void ClassTypeMapper::fillStringClassesVector(const std::string &classesFile) {
    std::ifstream labelFile(classesFile);
    std::string data;
    while(getline(labelFile,data)) {
        this->classes.push_back(data);
    }
}

bool ClassTypeMapper::mapString(std::string &className) {
    std::vector<std::string>::iterator it;

    it = find (this->classes.begin(), this->classes.end(), className);

    // For Open Images Dataset
    int i = 0;
    bool found = false;
    while (i < this->classes.size() && !found) {
	std::string splittedClass = this->classes[i].substr(this->classes[i].find(",") + 1, this->classes[i].size()-2);
	std::transform(splittedClass.begin(), splittedClass.end(), splittedClass.begin(), [](unsigned char c){ return std::tolower(c); });
	splittedClass = splittedClass.substr(0, splittedClass.size()-1);

	if (splittedClass == className) {
	    found = true;
	    this->classID = this->classes[i];
	    return true;
	}
	i++;
    }

    if (it != this->classes.end()) {
        this->classID = className;
        return true;                    //Class Name already present in dataset file
    }

    std::vector<std::string> syns = this->root.getImmediateSynonmys(className);
    std::vector<std::string>::iterator itr;

    if (!syns.empty()) {
        for (itr = syns.begin(); itr != syns.end(); itr++) {
            it = find (this->classes.begin(), this->classes.end(), *itr);
            if (it != this->classes.end()) {
                this->classID = *itr;
                return true;
            }
        }
    }

    return false;
}

std::unordered_map<std::string, std::string> ClassTypeMapper::mapFile(std::string classNamesFile) {

    std::unordered_map<std::string, std::string> classMap;

    std::ifstream myReadFile;
    myReadFile.open(classNamesFile);
    std::string output;

    if (myReadFile.is_open()) {
        while (getline(myReadFile, output)) {
            if (mapString(output))
                classMap.insert(std::pair<std::string,std::string>(output, this->classID));
            else
                classMap.insert(std::pair<std::string,std::string>(output, ""));
             //std::cout << output << '\n'; // Prints our STRING.

        }

    }

    myReadFile.close();

    return classMap;

}
