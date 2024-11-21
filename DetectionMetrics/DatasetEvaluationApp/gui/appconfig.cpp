#include "appconfig.h"
#include "ui_appconfig.h"
#include <QFileDialog>
#include <QMessageBox>
#include <iostream>

// Constructor to initialize and configure all the buttons
appconfig::appconfig(QWidget *parent) : QMainWindow(parent), ui(new Ui::appconfig){
    ui->setupUi(this);
    this->node;
    // Connect all the buttons to listen to certain actions and
    // trigger callback functions if action performed.
    connect(ui->toolButton_weights, SIGNAL (clicked()),this, SLOT (handleToolbuttonWeights()));
    connect(ui->toolButton_eval, SIGNAL (clicked()),this, SLOT (handleToolbuttonEval()));
    connect(ui->toolButton_cfg, SIGNAL (clicked()),this, SLOT (handleToolbuttonCfg()));
    connect(ui->toolButton_appconfig, SIGNAL (clicked()),this, SLOT (handleToolbuttonAppconfig()));
    connect(ui->toolButton_names, SIGNAL (clicked()),this, SLOT (handleToolbuttonNames()));
    connect(ui->pushButton_ok, SIGNAL (clicked()),this, SLOT (handlePushbuttonOK()));
    connect(ui->checkBox, SIGNAL (clicked()),this, SLOT (handleCheckbox()));
    this->node["datasetPath"]="~/";
}

// To select an config file
void appconfig::handleToolbuttonAppconfig(){
    QString dir_name = QFileDialog::getOpenFileName(this,"Select config file","~/");
    ui->lineEdit_appconfig->setText(dir_name);
   this->node["appconfig"]=dir_name.toUtf8().constData();
}

// To select weightsPath
void appconfig::handleToolbuttonWeights(){
    QString dir_name = QFileDialog::getExistingDirectory(this,"Open a dir","~/");
    ui->lineEdit_weights->setText(dir_name);
   this->node["weightsPath"]=dir_name.toUtf8().constData();
}

// To select config path
void appconfig::handleToolbuttonCfg(){
    QString dir_name = QFileDialog::getExistingDirectory(this,"Open a dir","~/");
    ui->lineEdit_cfg->setText(dir_name);
   this->node["netCfgPath"]=dir_name.toUtf8().constData();
}

// To select NamesDir path
void appconfig::handleToolbuttonNames(){
    QString dir_name = QFileDialog::getExistingDirectory(this,"Open a dir","~/");
    ui->lineEdit_names->setText(dir_name);
    this->node["namesPath"]=dir_name.toUtf8().constData();
}

// To select evalutaion path
void appconfig::handleToolbuttonEval(){
    QString dir_name = QFileDialog::getExistingDirectory(this,"Open a dir","~/");
    ui->lineEdit_eval->setText(dir_name);
    this->node["evaluationsPath"]=dir_name.toUtf8().constData();
}

// Function to proceed forward if all the required parameters are passed
void appconfig::handlePushbuttonOK(){
  // Pop an error message if not all the parameters/or a config file is passed
   if(!ui->checkBox->isChecked()  && !ui->lineEdit_appconfig->text().size()){
      QMessageBox::warning(this,"AppConfig","Please select the AppConfig file or "
                                            "provide the below required parameters individually ");
      return;
   }
   // Similar check as above
   if(ui->checkBox->isChecked()){
        if( !ui->lineEdit_cfg->text().size() || ! ui->lineEdit_names->text().size() ||
                !ui->lineEdit_eval->text().size() || ! ui->lineEdit_weights->text().size() ){
            QMessageBox::warning(this,"AppConfig","Please provide the required parameters to proceed");
            return;
        }
   }

   // If everything runs smoothly exit
    QApplication::quit();
    QCoreApplication::quit();
    // return ;
}

// Return a Yaml node
YAML::Node appconfig::return_node(){
 return this->node;
}

// Graying out not required parameters
void appconfig::handleCheckbox(){
        ui->lineEdit_weights->setDisabled(!ui->checkBox->isChecked());
        ui->lineEdit_names->setDisabled(!ui->checkBox->isChecked());
        ui->lineEdit_eval->setDisabled(!ui->checkBox->isChecked());
        ui->lineEdit_cfg->setDisabled(!ui->checkBox->isChecked());
        ui->toolButton_weights->setDisabled(!ui->checkBox->isChecked());
        ui->toolButton_names->setDisabled(!ui->checkBox->isChecked());
        ui->toolButton_eval->setDisabled(!ui->checkBox->isChecked());
        ui->toolButton_cfg->setDisabled(!ui->checkBox->isChecked());
}

// Destructor
appconfig::~appconfig(){
    delete ui;
}
