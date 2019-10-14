#include "mainwindow.h"
#include <QApplication>
#include <Utils/SampleGenerationApp.h>
#include <QStyleFactory>
#include "gui/Appcfg.hpp"

class MyApp:public SampleGenerationApp{
public:
  // Constructor Functions were written this form to avoid segmentation fault
  MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
      this->requiredArguments.push_back("datasetPath");
      this->requiredArguments.push_back("evaluationsPath");
      this->requiredArguments.push_back("weightsPath");
      this->requiredArguments.push_back("netCfgPath");
      this->requiredArguments.push_back("namesPath");
  };
  MyApp(YAML::Node node):SampleGenerationApp(node){
      this->requiredArguments.push_back("datasetPath");
      this->requiredArguments.push_back("evaluationsPath");
      this->requiredArguments.push_back("weightsPath");
      this->requiredArguments.push_back("netCfgPath");
      this->requiredArguments.push_back("namesPath");
  };
  MyApp(std::string filepath, bool isPath):SampleGenerationApp(filepath,isPath){
      this->requiredArguments.push_back("datasetPath");
      this->requiredArguments.push_back("evaluationsPath");
      this->requiredArguments.push_back("weightsPath");
      this->requiredArguments.push_back("netCfgPath");
      this->requiredArguments.push_back("namesPath");
  };
    void operator()(){
        QApplication a(argc, argv);
        MainWindow w(this);
        w.show();
        a.exec();

    };
};




int main(int argc, char *argv[]){
  // Check how many arguments are passed

  if(argc<3){
    // If less than 3 , then pop up the gui.
    Appcfg app(argc,argv);
    YAML::Node noder = app.get_node();
    // Check if appconfig is passed
    if(noder["appconfig"]){
      // If yes , convert that to a string and run detection suite
      MyApp myApp(noder["appconfig"].as<std::string>(),true);
      myApp.process();
    }
    else{
      // Else pass that YAML node directly which requires no further checks by
      // SampleGenerationApp.
      MyApp myApp(noder);
      myApp.process();
    }
  }
  else{
    // If a config file is passed , rest is handled by SampleGenerationApp
    MyApp myApp(argc,argv);
    myApp.process();
  }
}
