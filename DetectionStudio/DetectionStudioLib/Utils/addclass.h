#ifndef ADDCLASS_H
#define ADDCLASS_H

#include <iostream>
#include <QMainWindow>
#include <QEventLoop>

namespace Ui {
class AddClass;
}

class AddClass : public QMainWindow
{
    Q_OBJECT

public:
    explicit AddClass(QWidget *parent = 0);
    ~AddClass();
    void SetInit(std::vector<std::string> *classNames,std::string *name_f,double *probability);
    void wait();

private slots:
    void HandlePushButton_ok();
    void HandlePushButton_cancel();
    void HandleCheckbox();

private:
    Ui::AddClass *ui;
    // Final name
    std::string *name_f;
    // Final probability
    double *probability;
};

#endif // ADDCLASS_H
