#include "addclass.h"
#include "ui_addclass.h"

// Classic constructor function , where all the buttons are given some action to
// check for , then the corresponding callback functions are called.
AddClass::AddClass(QWidget *parent) : QMainWindow(parent),ui(new Ui::AddClass){
    ui->setupUi(this);
    connect(ui->pushButton_ok    , SIGNAL (clicked()),this, SLOT (HandlePushButton_ok()));
    connect(ui->pushButton_cancel, SIGNAL (clicked()),this, SLOT (HandlePushButton_cancel()));
    connect(ui->checkBox, SIGNAL (clicked()),this, SLOT (HandleCheckbox()));
    AddClass::HandleCheckbox();
}

// Destructor
AddClass::~AddClass(){
    delete ui;
}

// Don't exit until all the required parameters are
void AddClass::HandlePushButton_ok(){
    *(this->name_f) =  ui->checkBox->isChecked() ? ui->comboBox->currentText().toUtf8().constData() :
                                                   ui->lineEdit->text().toUtf8().constData() ;
    *(this->probability) = ui->probability->text().toDouble();
   if(!this->name_f->length() || !ui->probability->text().length())
     return;
    delete this;
}

// Delete this Q_OBJECT
void AddClass::HandlePushButton_cancel(){
    delete this;
}

// load the classes from the classNames files and other parameters like probability and final names
void AddClass::SetInit(std::vector<std::string>*classNames,std::string *name_f,double *probability){
    for(unsigned int i=0;i<classNames->size();i++)
        ui->comboBox->addItem(QString::fromStdString(classNames->at(i)));
    this->name_f=name_f;
    this->probability = probability;
}

// Wait untill the user finish interacting with user
void AddClass::wait(){
  QEventLoop loop;
  connect(this, SIGNAL(destroyed()), &loop, SLOT(quit()));
  loop.exec();
}

void AddClass::HandleCheckbox(){
    ui->comboBox->setDisabled(!ui->checkBox->isChecked());
    ui->lineEdit->setDisabled(ui->checkBox->isChecked());
}
