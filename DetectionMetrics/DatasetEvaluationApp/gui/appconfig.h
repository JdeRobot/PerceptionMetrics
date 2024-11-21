#ifndef APPCONFIG_H
#define APPCONFIG_H

// This is the backend to select the required parameters graphically

#include <QMainWindow>
#include <yaml-cpp/yaml.h>

namespace Ui {
class appconfig;
}

class appconfig : public QMainWindow
{
    Q_OBJECT

public:
    explicit appconfig(QWidget *parent = 0); // Constructor
    ~appconfig(); // Destructor
    YAML::Node return_node(); // Returns YAML node
private slots:
  // Callback functions to handle different buttons
    void handleToolbuttonWeights();
    void handleToolbuttonNames();
    void handleToolbuttonCfg();
    void handleToolbuttonAppconfig();
    void handleToolbuttonEval();
    void handleCheckbox();
    void handlePushbuttonOK();

private:
    Ui::appconfig *ui;
    YAML::Node node;
};

#endif // APPCONFIG_H
