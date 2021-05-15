#include "setclass.h"
#include "ui_setclass.h"

// Classic constructor function , where all the buttons are given some action to
// check for , then the corresponding callback functions are called.
SetClass::SetClass(QWidget *parent) : QMainWindow(parent),ui(new Ui::SetClass){
    ui->setupUi(this);
    connect(ui->pushButton_ok    , SIGNAL (clicked()),this, SLOT (HandlePushButton_ok()));
    connect(ui->pushButton_cancel, SIGNAL (clicked()),this, SLOT (HandlePushButton_cancel()));
}

// Destructor
SetClass::~SetClass(){
    delete ui;
}

// Set the selected class name and delete "this" Q_OBJECT
void SetClass::HandlePushButton_ok(){
    *(this->name_f)= ui->comboBox->currentText().toUtf8().constData();
    delete this;
}

// Delete this Q_OBJECT
void SetClass::HandlePushButton_cancel(){
    delete this;
}

// Set the current class name of the file which will be changed later by the user.
void SetClass::SetInit(std::string *str , std::vector<std::string>*classNames,std::string *name_f){
    ui->lineEdit->setText(QString::fromStdString(*str));
    // Loop through all the avialable classes present in the classnames file
    for(unsigned int i=0;i<classNames->size();i++)
        ui->comboBox->addItem(QString::fromStdString(classNames->at(i)));

    this->name_f=name_f;
}

// Wait until the user selects a class, i.e stop everything else , including DetectionsSuite
void SetClass::wait(){
  QEventLoop loop;
  connect(this, SIGNAL(destroyed()), &loop, SLOT(quit()));
  loop.exec();
}
