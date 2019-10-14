#ifndef APP_CONFIG
#define APP_CONFIG

// This is just for initialization of things required to start GUI
// For more refer to "appconfig" class
#include "appconfig.h"
#include <QApplication>

class Appcfg {
public:
  // constructor
  Appcfg(int argc , char **argv);
  // To start the GUI
  void exec();
  // Yaml node that stores all the required parameters to start DetectionSuite
  YAML::Node get_node();

private:
  QApplication *a;
  appconfig *w;
};

#endif
// int main(int argc, char *argv[])
// {
//   QApplication a(argc, argv);
//   MainWindow w;
//   w.show();
//   return a.exec();
// }
