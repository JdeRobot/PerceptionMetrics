#include "Appcfg.hpp"
#include <iostream>

// Classic constructor
Appcfg::Appcfg(int argc, char **argv){
    this->a = new QApplication(argc,argv);
    this->w = new appconfig();
    Appcfg::exec();
}

// Starts the GUI
void Appcfg::exec(){
  this->w->show();
  this->a->exec();
}

/* Returns the YAML node which has information regarding the required parameters
 to start the suite */
YAML::Node Appcfg::get_node(){
  return this->w->return_node();
}
