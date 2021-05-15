#ifndef SETCLASS_H
#define SETCLASS_H
// This is used to change the class names of the wrongly detected classes.
#include <iostream>
#include <QMainWindow>
#include <QEventLoop>

namespace Ui {
class SetClass;
}

class SetClass : public QMainWindow{
    Q_OBJECT

public:
    explicit SetClass(QWidget *parent = 0);
    ~SetClass();
    void SetInit(std::string *name, std::vector<std::string> *classNames,std::string *name_f);
    void wait();

private slots:
    void HandlePushButton_ok();
    void HandlePushButton_cancel();

private:
    Ui::SetClass *ui;
    std::string *name_f;
};

#endif // SETCLASS_H
