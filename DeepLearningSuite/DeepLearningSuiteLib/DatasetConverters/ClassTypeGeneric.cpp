//
// Created by frivas on 9/02/17.
//

#include <fstream>
#include "ClassTypeGeneric.h"


/*
    Constructor function when id is not given.
    Only a class vector containing the classes
    in the file is created.
*/
ClassTypeGeneric::ClassTypeGeneric(const std::string &classesFile) {
  // Create a vector of strings with classes in it.
    fillStringClassesVector(classesFile);
}

/*
   Constructor function when id is given.
   Not only a class vector is created but
   also the classID is initialized.
*/
ClassTypeGeneric::ClassTypeGeneric(const std::string &classesFile, int id) {
  // Create a vector of strings with classes in it.
    fillStringClassesVector(classesFile);
    this->classID=this->classes[id];
}


/*
  Loop over all the classes available in the classesFile and
  store them in vector of strings, named "classes".
*/
void ClassTypeGeneric::fillStringClassesVector(const std::string &classesFile) {
    std::ifstream labelFile(classesFile);
    std::string data;
    while(getline(labelFile,data)) {
        this->classes.push_back(data);
    }
}

// Update/Set the classID
void ClassTypeGeneric::setId(int id) {
    this->classID=this->classes[id];
}

// Print all the classes 
void ClassTypeGeneric::Print(){
  for(auto itr=this->classes.begin();itr!=this->classes.end();itr++)
      std::cout << *itr << std::endl;
}
