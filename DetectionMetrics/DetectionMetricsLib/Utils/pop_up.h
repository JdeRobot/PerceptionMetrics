#ifndef POP_UP_H
#define POP_UP_H

// This is triggered if any of the required parameters are missing from the config file

#include <QApplication>
#include "QMainWindow"

#include <iostream>
#include <yaml-cpp/yaml.h>

namespace Ui {
class pop_up;
}

class pop_up : public QMainWindow
{
    Q_OBJECT

public:
    explicit pop_up(QWidget *parent = 0);
    void SetName(std::string Name);
    void SetPath(std::string *path);
    ~pop_up();
private slots:
    void HandleToolButton_1();
    void HandlePushButton_ok();

private:
    Ui::pop_up *ui;
    YAML::Node node;
    QString name;
    std::string *path;
};

#endif // POP_UP_H
