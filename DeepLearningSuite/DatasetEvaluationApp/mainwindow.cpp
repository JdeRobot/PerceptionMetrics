#include <QtWidgets/QListWidget>
#include <QtCore/QStringListModel>
#include <gui/ListViewConfig.h>
#include <SamplerGeneratorHandler/Viewer.h>
#include <SamplerGeneratorHandler/Converter.h>
#include <DatasetConverters/readers/GenericDatasetReader.h>
#include <DatasetConverters/ClassTypeGeneric.h>
#include <glog/logging.h>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QTreeView>
#include <DatasetConverters/writers/GenericDatasetWriter.h>
#include <gui/Utils.h>
#include <SamplerGeneratorHandler/Evaluator.h>
#include <SamplerGeneratorHandler/Detector.h>
#include <FrameworkEvaluator/GenericInferencer.h>
#include <DatasetConverters/liveReaders/GenericLiveReader.h>
#include <SamplerGeneratorHandler/Deployer.h>
#include "mainwindow.h"
#include "ui_mainwindow.h"
// #include "gui/Appcfg.hpp"

MainWindow::MainWindow(SampleGenerationApp* app,QWidget *parent) :
    app(app),
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    setupTabsInformation();




    connect(ui->pushButton, SIGNAL (released()),this, SLOT (handleViewButton()));
    connect(ui->tabWidget, SIGNAL(currentChanged(int)), this, SLOT(setupTabsInformation()));
    connect(ui->pushButton_converter_output, SIGNAL (released()),this, SLOT (handleSelectOutputFolderButton()));
    connect(ui->pushButton_detector_output, SIGNAL (released()),this, SLOT (handleSelectOutputFolderButtonDetector()));
    connect(ui->pushButton_convert, SIGNAL (released()),this, SLOT (handleConvertButton()));
    connect(ui->pushButton_evaluate, SIGNAL (released()),this, SLOT (handleEvaluateButton()));
    connect(ui->pushButton_detect, SIGNAL (released()),this, SLOT (handleDetectButton()));
    connect(ui->pushButton_deploy_input, SIGNAL (released()),this, SLOT (handleSelectDeployInputSource()));
    connect(ui->pushButton_deploy_process, SIGNAL (released()),this, SLOT (handleProcessDeploy()));
    connect(ui->checkBox_deployer_saveOutput, SIGNAL (released()), this, SLOT( handleDeployerSaveOutputCheckboxChange()));
    // connect(ui->checkBox_deployer_saveOutput, SIGNAL (released()), this, SLOT( handleDeployerSaveOutputCheckboxChange()));
    connect(ui->pushButton_stop_deployer_process, SIGNAL(released()), this, SLOT(handleDeployerStop()));
    connect(ui->pushButton_deployer_output_folder, SIGNAL(released()), this, SLOT(handleSelectOutputFolderButtonDeployer()));
    connect(ui->deployer_conf_horizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(handleDeployerConfidenceSliderChange(int)));
    connect(ui->deployer_confidence_lineEdit, SIGNAL(textEdited(QString)), this, SLOT(handleDeployerConfidenceLineEditChange(QString)));


}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::handleViewButton() {
    SampleGeneratorHandler::Viewer::process(ui->listView_viewer_dataset,ui->listView_viewer_names,ui->listView_viewer_reader_imp,
                                            ui->listView_viewer_classFilter, ui->checkBox_evaluator_show_depth->isChecked(), app->getConfig().asString("datasetPath"), app->getConfig().asString("namesPath"));
}

void MainWindow::handleSelectionNamesChanged() {
    std::string classNameFilePath;
    LOG(INFO) << ui->tabWidget->currentIndex() << std::endl;
    switch(ui->tabWidget->currentIndex()) {
        case 0: {
            std::vector<std::string> dataSelected;
            Utils::getListViewContent(ui->listView_viewer_names,dataSelected,app->getConfig().asString("namesPath") + "/");
            ClassTypeGeneric typeConverter(dataSelected[0]);
            ListViewConfig::configureInputByData(this, ui->listView_viewer_classFilter,
                                                 typeConverter.getAllAvailableClasses(), true);
            }
            break;
        case 1: {
            std::vector<std::string> dataSelected;
            Utils::getListViewContent(ui->listView_converter_names,dataSelected,app->getConfig().asString("namesPath") + "/");
            ClassTypeGeneric typeConverter(dataSelected[0]);
            ListViewConfig::configureInputByData(this, ui->listView_converter_classFilter,
                                                 typeConverter.getAllAvailableClasses(), true);
            }
            break;
        case 3: {
            std::vector<std::string> dataSelected;
            Utils::getListViewContent(ui->listView_evaluator_detection_names, dataSelected,
                                      app->getConfig().asString("namesPath") + "/");
            ClassTypeGeneric typeConverter(dataSelected[0]);
            ListViewConfig::configureInputByData(this, ui->listView_evaluator_classFilter,
                                                 typeConverter.getAllAvailableClasses(), true);
            break;
        }
        case 4:{

            break;
            }
        default:
            LOG(WARNING) << "Unkown tab index";
    }
}

void MainWindow::handleMappingCheckBoxChange() {
    if(ui->checkBox_use_writernames->isChecked()) {
        ui->listView_converter_writer_names->setEnabled(true);
    } else {
        ui->listView_converter_writer_names->setEnabled(false);
    }
}

void MainWindow::setupTabsInformation() {
    switch(ui->tabWidget->currentIndex()) {
        case 0:
            ListViewConfig::configureDatasetInput(this, ui->listView_viewer_dataset,
                                              app->getConfig().asString("datasetPath"), true);
            ListViewConfig::configureInputByFile(this, ui->listView_viewer_names,
                                             app->getConfig().asString("namesPath"), false);
            ListViewConfig::configureInputByData(this, ui->listView_viewer_reader_imp,
                                                 GenericDatasetReader::getAvailableImplementations(), false);
            connect(ui->listView_viewer_names->selectionModel(), SIGNAL(selectionChanged(QItemSelection,QItemSelection)), this, SLOT(handleSelectionNamesChanged()));

            break;
        case 1:
            ListViewConfig::configureDatasetInput(this, ui->listView_converter_dataset,
                                                  app->getConfig().asString("datasetPath"), true);
            ListViewConfig::configureInputByFile(this, ui->listView_converter_names,
                                                 app->getConfig().asString("namesPath"), false);
            ListViewConfig::configureInputByFile(this, ui->listView_converter_writer_names,
                                              app->getConfig().asString("namesPath"), false);
            ui->listView_converter_writer_names->setEnabled(false);
            ListViewConfig::configureInputByData(this, ui->listView_converter_reader_imp,
                                                 GenericDatasetReader::getAvailableImplementations(), false);
            ListViewConfig::configureInputByData(this, ui->listView_converter_outImp,
                                                 GenericDatasetWriter::getAvailableImplementations(), false);

            connect(ui->listView_converter_names->selectionModel(), SIGNAL(selectionChanged(QItemSelection,QItemSelection)), this, SLOT(handleSelectionNamesChanged()));
            connect(ui->checkBox_use_writernames, SIGNAL(clicked(bool)), this, SLOT(handleMappingCheckBoxChange()));
            break;
        case 2:
            ListViewConfig::configureDatasetInput(this, ui->listView_detector_dataset,
                                                  app->getConfig().asString("datasetPath"), true);
            ListViewConfig::configureInputByFile(this, ui->listView_detector_names,
                                                 app->getConfig().asString("namesPath"), false);
            ListViewConfig::configureInputByData(this, ui->listView_detector_reader_imp,
                                                 GenericDatasetReader::getAvailableImplementations(), false);
            ListViewConfig::configureInputByFile(this, ui->listView_detector_weights,
                                                 app->getConfig().asString("weightsPath"), false);
            ListViewConfig::configureInputByFile(this, ui->listView_detector_net_config,
                                                 app->getConfig().asString("netCfgPath"), false);
            ListViewConfig::configureInputByData(this, ui->listView_detector_imp,
                                                 GenericInferencer::getAvailableImplementations(), false);
            ListViewConfig::configureInputByFile(this, ui->listView_detector_names_inferencer,
                                                 app->getConfig().asString("namesPath"), false);


            ui->detector_groupBox_inferencer_params->setEnabled(false);

            connect(ui->listView_detector_imp->selectionModel(),SIGNAL(currentRowChanged(QModelIndex,QModelIndex)), this,SLOT(handleDetectorInferencerImpListViewChange(QModelIndex, QModelIndex)));



            break;
        case 3:
            ListViewConfig::configureDatasetInput(this, ui->listView_evaluator_gt_dataset,
                                                  app->getConfig().asString("datasetPath"), true);
            ListViewConfig::configureInputByFile(this, ui->listView_evaluator_gt_names,
                                                 app->getConfig().asString("namesPath"), false);
            ListViewConfig::configureInputByData(this, ui->listView_evaluator_gt_imp,
                                                 GenericDatasetReader::getAvailableImplementations(), false);
            ListViewConfig::configureDatasetInput(this, ui->listView_evaluator_dectection_dataset,
                                                  app->getConfig().asString("inferencesPath"), true);
            ListViewConfig::configureInputByFile(this, ui->listView_evaluator_detection_names,
                                                 app->getConfig().asString("namesPath"), false);
            ListViewConfig::configureInputByData(this, ui->listView_evaluator_detection_imp,
                                                 GenericDatasetReader::getAvailableImplementations(), false);
            connect(ui->listView_evaluator_detection_names->selectionModel(), SIGNAL(selectionChanged(QItemSelection,QItemSelection)), this, SLOT(handleSelectionNamesChanged()));
            break;
        case 4:
            ListViewConfig::configureInputByFile(this, ui->listView_deploy_weights,
                                                 app->getConfig().asString("weightsPath"), false);
            ListViewConfig::configureInputByFile(this, ui->listView_deploy_net_config,
                                                 app->getConfig().asString("netCfgPath"), false);
            ListViewConfig::configureInputByData(this, ui->listView_deploy_impl,
                                                 GenericInferencer::getAvailableImplementations(), false);
            ListViewConfig::configureInputByFile(this, ui->listView_deploy_names_inferencer,
                                                 app->getConfig().asString("namesPath"), false);
            ListViewConfig::configureInputByData(this, ui->listView_deploy_input_imp,
                                                 GenericLiveReader::getAvailableImplementations(), false);


            ui->deployer_param_groupBox->setEnabled(false);
            ui->groupBox_config_option->setEnabled(false);
            ui->deployer_radioButton_manual->setChecked(true);

            #ifdef ICE
            ui->radioButton_deployer_ice->setChecked(true);
            #else
            ui->radioButton_deployer_ice->setEnabled(false);
            #endif
            #ifdef JDERROS
            ui->radioButton_deployer_ros->setChecked(true);
            #else
            ui->radioButton_deployer_ros->setEnabled(false);
            #endif

            ui->deployer_groupBox_inferencer_params->setEnabled(false);
            ui->deployer_cameraID_groupBox->setEnabled(false);

            connect(ui->listView_deploy_input_imp->selectionModel(),SIGNAL(currentRowChanged(QModelIndex,QModelIndex)), this,SLOT(handleDeployerImpListViewChange(QModelIndex, QModelIndex)));
            //connect(ui->groupBox_config_file, SIGNAL(toggled(bool)), this, SLOT(handleDeployerConfigFileOptionChange(bool)));
            connect(ui->deployer_radioButton_manual, SIGNAL(toggled(bool)), this, SLOT(handleDeployerConfigFileOptionChange(bool)));
            connect(ui->listView_deploy_impl->selectionModel(),SIGNAL(currentRowChanged(QModelIndex,QModelIndex)), this,SLOT(handleDeployerInferencerImpListViewChange(QModelIndex, QModelIndex)));



            break;
        default:
            LOG(WARNING) << "Unkown tab index";
    }
}

void MainWindow::handleSelectOutputFolderButton() {
    QFileDialog *fd = new QFileDialog;
    QTreeView *tree = fd->findChild <QTreeView*>();
#ifndef __APPLE__
    tree->setRootIsDecorated(true);
    tree->setItemsExpandable(false);
#endif
    fd->setFileMode(QFileDialog::Directory);
    fd->setOption(QFileDialog::ShowDirsOnly);
    fd->setViewMode(QFileDialog::Detail);
    int result = fd->exec();
    QString directory;
    if (result)
    {
        directory = fd->selectedFiles()[0];
        this->ui->textEdit_converterOutPath->setText(directory);
    }
}

void MainWindow::handleConvertButton() {
    double ratio;
    ratio = this->ui->textEdit_converter_trainRatio->toPlainText().toDouble();
    std::string outputPath = this->ui->textEdit_converterOutPath->toPlainText().toStdString();
    bool splitActive = this->ui->checkBox_splitActive->isChecked();
    bool writeImages = this->ui->checkBox_converter_write_images->isChecked();

    try {
        SampleGeneratorHandler::Converter::process(ui->listView_converter_dataset, ui->listView_converter_names,
                                                   ui->listView_converter_reader_imp,
                                                   ui->listView_converter_classFilter,
                                                   ui->listView_converter_outImp,
                                                   ui->listView_converter_writer_names,
                                                   ui->checkBox_use_writernames->isChecked(),
                                                   app->getConfig().asString("datasetPath"),
                                                   app->getConfig().asString("namesPath"), outputPath,
                                                   splitActive, ratio, writeImages);
    }
    catch (const std::string& msg){
        LOG(ERROR) << "Exception detected: " << msg << std::endl;
    }
    catch (const std::exception &exc)
    {
        LOG(ERROR) << "Exception Detected: " << exc.what();
    }
    catch (...){
        LOG(ERROR) << "Uknown exception type" << std::endl;
    }
}

void MainWindow::handleEvaluateButton() {
    try{
    SampleGeneratorHandler::Evaluator::process(ui->listView_evaluator_gt_dataset,ui->listView_evaluator_gt_names,ui->listView_evaluator_gt_imp,
                                               ui->listView_evaluator_dectection_dataset,ui->listView_evaluator_detection_names, ui->listView_evaluator_detection_imp,
                                               ui->listView_evaluator_classFilter,app->getConfig().asString("datasetPath"),app->getConfig().asString("namesPath"),
                                               app->getConfig().asString("inferencesPath"),app->getConfig().asString("namesPath"),ui->checkBox_evaluator_merge->isChecked(),
                                               ui->checkBox_evaluator_mix->isChecked(), ui->radioButton_evaluator_iou_bbox->isChecked());
    }
    catch (const std::string& msg){
        LOG(ERROR) << "Exception detected: " << msg;
    }
    catch (const std::exception &exc)
    {
        LOG(ERROR) << "Exeption Detected: " << exc.what();
    }
    catch (...){
        LOG(ERROR) << "Uknown Exception";
    }
}

void MainWindow::handleDetectButton() {
    std::string outputPath = this->ui->textEdit_detectorOutPath->toPlainText().toStdString();
    bool useDepth = this->ui->checkBox_detector_useDepth->isChecked();
    bool singleEvaluation = this->ui->checkBox_detector_single->isChecked();
    QGroupBox* inferencer_params = this->ui->detector_groupBox_inferencer_params;


    try{
    SampleGeneratorHandler::Detector::process(ui->listView_detector_dataset, ui->listView_detector_names,ui->listView_detector_reader_imp,app->getConfig().asString("datasetPath"),
                                              ui->listView_detector_weights,ui->listView_detector_net_config,ui->listView_detector_imp,ui->listView_detector_names_inferencer,
                                              inferencer_params, app->getConfig().asString("weightsPath"),app->getConfig().asString("netCfgPath"),outputPath,app->getConfig().asString("namesPath"),
                                              useDepth,singleEvaluation);
    }
    catch (const std::string& msg){
        LOG(ERROR) << "Exception Detected: " << msg;
    }
    catch (const std::exception &exc)
    {
        LOG(ERROR) << "Exeption Detected: " << exc.what();
    }
    catch (...){
        LOG(ERROR) << "Uknown exectip Type";
    }
}

void MainWindow::handleSelectOutputFolderButtonDetector() {
    QFileDialog *fd = new QFileDialog;
    QTreeView *tree = fd->findChild <QTreeView*>();
#ifndef __APPLE__
    tree->setRootIsDecorated(true);
    tree->setItemsExpandable(false);
#endif
    fd->setFileMode(QFileDialog::Directory);
    fd->setOption(QFileDialog::ShowDirsOnly);
    fd->setViewMode(QFileDialog::Detail);
    int result = fd->exec();
    QString directory;
    if (result)
    {
        directory = fd->selectedFiles()[0];
        this->ui->textEdit_detectorOutPath->setText(directory);
    }
}

void MainWindow::handleSelectDeployInputSource() {
    QFileDialog *fd = new QFileDialog;
    QTreeView *tree = fd->findChild <QTreeView*>();
#ifndef __APPLE__
    tree->setRootIsDecorated(true);
    tree->setItemsExpandable(false);
#endif
    fd->setFileMode(QFileDialog::AnyFile);
//    fd->setOption(QFileDialog::Show);
    fd->setViewMode(QFileDialog::Detail);
    int result = fd->exec();
    QString directory;
    if (result)
    {
        directory = fd->selectedFiles()[0];
        this->ui->textEdit_deployInputPath->setText(directory);
    }
}

void MainWindow::handleSelectOutputFolderButtonDeployer() {
    QFileDialog *fd = new QFileDialog;
    QTreeView *tree = fd->findChild <QTreeView*>();
#ifndef __APPLE__
    tree->setRootIsDecorated(true);
    tree->setItemsExpandable(false);
#endif
    fd->setFileMode(QFileDialog::Directory);
    //fd->setOption(QFileDialog::ShowDirsOnly);
    fd->setViewMode(QFileDialog::Detail);
    int result = fd->exec();
    QString directory;
    if (result)
    {
        directory = fd->selectedFiles()[0];
        this->ui->textEdit_deployerOutputPath->setText(directory);
    }
}

void MainWindow::handleDeployerImpListViewChange(const QModelIndex& selected, const QModelIndex& deselected) {
    if (selected.data().toString() == "stream") {
        ui->deployer_param_groupBox->setEnabled(true);
        ui->groupBox_config_option->setEnabled(true);
        ui->deployer_cameraID_groupBox->setEnabled(false);
        handleDeployerConfigFileOptionChange(ui->deployer_radioButton_manual->isChecked());
    } else if (selected.data().toString() == "camera") {
        ui->textEdit_deployInputPath->setEnabled(false);
        ui->pushButton_deploy_input->setEnabled(false);
        ui->deployer_param_groupBox->setEnabled(false);
        ui->groupBox_config_option->setEnabled(false);
        ui->deployer_cameraID_groupBox->setEnabled(true);
    }
    else {
        ui->textEdit_deployInputPath->setEnabled(true);
        ui->pushButton_deploy_input->setEnabled(true);
        ui->deployer_param_groupBox->setEnabled(false);
        ui->groupBox_config_option->setEnabled(false);
        ui->deployer_cameraID_groupBox->setEnabled(false);
    }
}

void MainWindow::handleDeployerConfigFileOptionChange(bool checked) {
    if(checked){
        ui->textEdit_deployInputPath->setEnabled(false);
        ui->pushButton_deploy_input->setEnabled(false);
        ui->deployer_param_groupBox->setEnabled(true);
   } else {
       ui->textEdit_deployInputPath->setEnabled(true);
       ui->pushButton_deploy_input->setEnabled(true);
       ui->deployer_param_groupBox->setEnabled(false);
   }
}

void MainWindow::handleDeployerInferencerImpListViewChange(const QModelIndex& selected, const QModelIndex& deselected) {
    if (selected.data().toString() == "caffe") {
        ui->deployer_groupBox_inferencer_params->setEnabled(true);
    } else {
        ui->deployer_groupBox_inferencer_params->setEnabled(false);
    }
}

void MainWindow::handleDetectorInferencerImpListViewChange(const QModelIndex& selected, const QModelIndex& deselected) {
    if (selected.data().toString() == "caffe") {
        ui->detector_groupBox_inferencer_params->setEnabled(true);
    } else {
        ui->detector_groupBox_inferencer_params->setEnabled(false);
    }
}

void MainWindow::handleDeployerConfidenceLineEditChange(const QString& confidence) {


    std::string conf_val = confidence.toStdString();

    //std::cout << conf_val << '\n';
    double val;

    try {

        val = std::stod(confidence.toStdString());
    } catch (...) {

        bool oldState = this->ui->deployer_conf_horizontalSlider->blockSignals(true);
        this->ui->deployer_conf_horizontalSlider->setValue(0);
        this->ui->deployer_conf_horizontalSlider->blockSignals(oldState);
        return;
    }
    //std::cout << val << '\n';
    if (val > 1.0) {
        QMessageBox::warning(this, QObject::tr("Confidence Threshold out of Bounds"), QObject::tr("Confidence Threshold can't be greater than 1.0, setting Threshold to 0.2"));
        val = 1.0;
        this->ui->deployer_confidence_lineEdit->setText(QString("0.2"));
    }
    if ( val < 0.0) {
        QMessageBox::warning(this, QObject::tr("Confidence Threshold out of Bounds"), QObject::tr("Confidence Threshold can't be smaller than 0, setting Threshold to 0.2"));
        val = 0;
        this->ui->deployer_confidence_lineEdit->setText(QString("0.2"));
    }
    bool oldState = this->ui->deployer_conf_horizontalSlider->blockSignals(true);
    this->ui->deployer_conf_horizontalSlider->setValue((int)(val*100));
    this->ui->deployer_conf_horizontalSlider->blockSignals(oldState);
    this->confidence_threshold = val;
}

void MainWindow::handleDeployerConfidenceSliderChange(const int& confidence) {

    std::stringstream str;
    double val = confidence/100.0;
    str << std::fixed << std::setprecision( 2 ) << val;
    QString qstr = QString::fromStdString(str.str());

    //std::cout << qstr.toStdString() << '\n';

    this->ui->deployer_confidence_lineEdit->setText(qstr);
    this->confidence_threshold = val;
}


void MainWindow::handleDeployerSaveOutputCheckboxChange() {
    if(ui->checkBox_deployer_saveOutput->isChecked()) {
        ui->groupbox_deployer_saveOutput->setEnabled(true);
    } else {
        ui->groupbox_deployer_saveOutput->setEnabled(false);
    }
}

void MainWindow::handleDeployerStop() {
    this->stopDeployer = true;
    LOG(WARNING) << "Stopping Deployer Process" << "\n";
}

void MainWindow::handleProcessDeploy() {
    this->stopDeployer = false;
    std::string inputInfo = this->ui->textEdit_deployInputPath->toPlainText().toStdString();

    QGroupBox* deployer_params = this->ui->deployer_param_groupBox;
    QGroupBox* camera_params = this->ui->deployer_cameraID_groupBox;
    QGroupBox* inferencer_params = this->ui->deployer_groupBox_inferencer_params;
    std::string outputFolder = this->ui->textEdit_deployerOutputPath->toPlainText().toStdString();
    if (!ui->checkBox_deployer_saveOutput->isChecked()) {
        outputFolder.clear();
    }

    try{
        // LOG(INFO) << " sad : " << ui->Labeling->isChecked() << std::endl;
        SampleGeneratorHandler::Deployer::process(ui->listView_deploy_input_imp,ui->listView_deploy_weights,
                                                  ui->listView_deploy_net_config,ui->listView_deploy_impl,ui->listView_deploy_names_inferencer, &this->stopDeployer,
                                                  &confidence_threshold, deployer_params, camera_params, inferencer_params, app->getConfig().asString("weightsPath"),
                                                  app->getConfig().asString("netCfgPath"),app->getConfig().asString("namesPath"),inputInfo, outputFolder,ui->Labeling->isChecked());
    }
    catch (const std::string& msg){
        LOG(ERROR) << "Exception detected: " << msg;
    }
    catch (const std::exception &exc)
    {
        LOG(ERROR) << "Exception Detected: " << exc.what();
    }
    catch (...){
        LOG(ERROR) << "Uknown Exception Type";
    }
}
