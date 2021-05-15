#include "pop_up.h"
#include "ui_pop_up.h"
#include <QFileDialog>
#include <QMessageBox>

// Classic constructor function , where all the buttons are given some action to
// check for , then the corresponding callback functions are called.
pop_up::pop_up(QWidget *parent) : QMainWindow(parent), ui(new Ui::pop_up){
    ui->setupUi(this);
    connect(ui->toolButton_1, SIGNAL (clicked()),this, SLOT (HandleToolButton_1()));
    connect(ui->pushButton_ok, SIGNAL (clicked()),this, SLOT (HandlePushButton_ok()));
}

// Destructor
pop_up::~pop_up()
{
    delete ui;
}

// Selecte the missing file
void pop_up::HandleToolButton_1(){
    QString dir_name = QFileDialog::getExistingDirectory(this,"Select config file","~/");
    ui->lineEdit->setText(dir_name);
    *(this->path) = dir_name.toUtf8().constData();
}

// If selected proceed further else  return warning
void pop_up::HandlePushButton_ok(){
     if(!ui->lineEdit->text().size()){
        QMessageBox::warning(this,"Warning","Please provide " + this->name +
                                              " parameter to continue");
        return;
     }
    QApplication::quit();
    QCoreApplication::quit();
}

void pop_up::SetName(std::string Name){
  // Convert from string to QString
    this->name = QString::fromStdString(Name);
    ui->label->setText(this->name);
}

// Set the path member variable to the selected path
void pop_up::SetPath(std::string *path){
  this->path = path;
}
