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
    //connect(ui->listView_deploy_input_imp, SIGNAL (indexesMoved(const QModelIndexList)), this, SLOT (handleDeployerImpListViewChange()));
    connect(ui->listView_deploy_input_imp->selectionModel(),SIGNAL(currentRowChanged(QModelIndex,QModelIndex)), this,SLOT(handleDeployerImpListViewChange(QModelIndex, QModelIndex)));
    //connect(ui->groupBox_config_file, SIGNAL(toggled(bool)), this, SLOT(handleDeployerConfigFileOptionChange(bool)));
    connect(ui->deployer_radioButton_manual, SIGNAL(toggled(bool)), this, SLOT(handleDeployerConfigFileOptionChange(bool)));
    connect(ui->listView_deploy_impl->selectionModel(),SIGNAL(currentRowChanged(QModelIndex,QModelIndex)), this,SLOT(handleDeployerInferencerImpListViewChange(QModelIndex, QModelIndex)));


}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::handleViewButton() {
    SampleGeneratorHandler::Viewer::process(ui->listView_viewer_dataset,ui->listView_viewer_names,ui->listView_viewer_reader_imp,
                                            ui->listView_viewer_classFilter, ui->checkBox_evaluator_show_depth->isChecked(), app->getConfig()->getKey("datasetPath").getValue(), app->getConfig()->getKey("namesPath").getValue());
}

void MainWindow::handleSelectionNamesChanged() {
    std::string classNameFilePath;
    std::cout << ui->tabWidget->currentIndex() << std::endl;
    switch(ui->tabWidget->currentIndex()) {
        case 0: {
            std::vector<std::string> dataSelected;
            Utils::getListViewContent(ui->listView_viewer_names,dataSelected,app->getConfig()->getKey("namesPath").getValue() + "/");
            ClassTypeGeneric typeConverter(dataSelected[0]);
            ListViewConfig::configureInputByData(this, ui->listView_viewer_classFilter,
                                                 typeConverter.getAllAvailableClasses(), true);
            }
            break;
        case 1: {
            std::vector<std::string> dataSelected;
            Utils::getListViewContent(ui->listView_converter_names,dataSelected,app->getConfig()->getKey("namesPath").getValue() + "/");
            ClassTypeGeneric typeConverter(dataSelected[0]);
            ListViewConfig::configureInputByData(this, ui->listView_converter_classFilter,
                                                 typeConverter.getAllAvailableClasses(), true);
            }
            break;
        case 3: {
            std::vector<std::string> dataSelected;
            Utils::getListViewContent(ui->listView_evaluator_detection_names, dataSelected,
                                      app->getConfig()->getKey("namesPath").getValue() + "/");
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
                                              app->getConfig()->getKey("datasetPath").getValue(), true);
            ListViewConfig::configureInputByFile(this, ui->listView_viewer_names,
                                             app->getConfig()->getKey("namesPath").getValue(), false);
            ListViewConfig::configureInputByData(this, ui->listView_viewer_reader_imp,
                                                 GenericDatasetReader::getAvailableImplementations(), false);
            connect(ui->listView_viewer_names->selectionModel(), SIGNAL(selectionChanged(QItemSelection,QItemSelection)), this, SLOT(handleSelectionNamesChanged()));

            break;
        case 1:
            ListViewConfig::configureDatasetInput(this, ui->listView_converter_dataset,
                                                  app->getConfig()->getKey("datasetPath").getValue(), true);
            ListViewConfig::configureInputByFile(this, ui->listView_converter_names,
                                                 app->getConfig()->getKey("namesPath").getValue(), false);
            ListViewConfig::configureInputByFile(this, ui->listView_converter_writer_names,
                                              app->getConfig()->getKey("namesPath").getValue(), false);
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
                                                  app->getConfig()->getKey("datasetPath").getValue(), true);
            ListViewConfig::configureInputByFile(this, ui->listView_detector_names,
                                                 app->getConfig()->getKey("namesPath").getValue(), false);
            ListViewConfig::configureInputByData(this, ui->listView_detector_reader_imp,
                                                 GenericDatasetReader::getAvailableImplementations(), false);
            ListViewConfig::configureInputByFile(this, ui->listView_detector_weights,
                                                 app->getConfig()->getKey("weightsPath").getValue(), false);
            ListViewConfig::configureInputByFile(this, ui->listView_detector_net_config,
                                                 app->getConfig()->getKey("netCfgPath").getValue(), false);
            ListViewConfig::configureInputByData(this, ui->listView_detector_imp,
                                                 GenericInferencer::getAvailableImplementations(), false);
            ListViewConfig::configureInputByFile(this, ui->listView_detector_names_inferencer,
                                                 app->getConfig()->getKey("namesPath").getValue(), false);

            break;
        case 3:
            ListViewConfig::configureDatasetInput(this, ui->listView_evaluator_gt_dataset,
                                                  app->getConfig()->getKey("datasetPath").getValue(), true);
            ListViewConfig::configureInputByFile(this, ui->listView_evaluator_gt_names,
                                                 app->getConfig()->getKey("namesPath").getValue(), false);
            ListViewConfig::configureInputByData(this, ui->listView_evaluator_gt_imp,
                                                 GenericDatasetReader::getAvailableImplementations(), false);
            ListViewConfig::configureDatasetInput(this, ui->listView_evaluator_dectection_dataset,
                                                  app->getConfig()->getKey("inferencesPath").getValue(), true);
            ListViewConfig::configureInputByFile(this, ui->listView_evaluator_detection_names,
                                                 app->getConfig()->getKey("namesPath").getValue(), false);
            ListViewConfig::configureInputByData(this, ui->listView_evaluator_detection_imp,
                                                 GenericDatasetReader::getAvailableImplementations(), false);
            connect(ui->listView_evaluator_detection_names->selectionModel(), SIGNAL(selectionChanged(QItemSelection,QItemSelection)), this, SLOT(handleSelectionNamesChanged()));
            break;
        case 4:
            ListViewConfig::configureInputByFile(this, ui->listView_deploy_weights,
                                                 app->getConfig()->getKey("weightsPath").getValue(), false);
            ListViewConfig::configureInputByFile(this, ui->listView_deploy_net_config,
                                                 app->getConfig()->getKey("netCfgPath").getValue(), false);
            ListViewConfig::configureInputByData(this, ui->listView_deploy_impl,
                                                 GenericInferencer::getAvailableImplementations(), false);
            ListViewConfig::configureInputByFile(this, ui->listView_deploy_names_inferencer,
                                                 app->getConfig()->getKey("namesPath").getValue(), false);
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

            ui->groupBox_deployer_inferencer_params->setEnabled(false);

            break;
        default:
            LOG(WARNING) << "Unkown tab index";
    }
}

void MainWindow::handleSelectOutputFolderButton() {
    QFileDialog *fd = new QFileDialog;
    QTreeView *tree = fd->findChild <QTreeView*>();
    tree->setRootIsDecorated(true);
    tree->setItemsExpandable(false);
    fd->setFileMode(QFileDialog::Directory);
    fd->setOption(QFileDialog::ShowDirsOnly);
    fd->setViewMode(QFileDialog::Detail);
    fd->setDirectory("/mnt/large/pentalo/deep/datasets/");
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
    bool colorImage = !(this->ui->checkBox_yolo_depth->isChecked());

    try {
        SampleGeneratorHandler::Converter::process(ui->listView_converter_dataset, ui->listView_converter_names,
                                                   ui->listView_converter_reader_imp,
                                                   ui->listView_converter_classFilter,
                                                   ui->listView_converter_outImp,
                                                   ui->listView_converter_writer_names,
                                                   ui->checkBox_use_writernames->isChecked(),
                                                   app->getConfig()->getKey("datasetPath").getValue(),
                                                   app->getConfig()->getKey("namesPath").getValue(), outputPath,
                                                   splitActive, ratio, colorImage);
    }
    catch (const std::string& msg){
        std::cout << "Exception detected: " << msg << std::endl;
    }
    catch (const std::exception &exc)
    {
        std::cout << "Exeption Detected: " << exc.what();
    }
    catch (...){
        std::cout << "Uknown exception type" << std::endl;
    }
}

void MainWindow::handleEvaluateButton() {
    try{
    SampleGeneratorHandler::Evaluator::process(ui->listView_evaluator_gt_dataset,ui->listView_evaluator_gt_names,ui->listView_evaluator_gt_imp,
                                               ui->listView_evaluator_dectection_dataset,ui->listView_evaluator_detection_names, ui->listView_evaluator_detection_imp,
                                               ui->listView_evaluator_classFilter,app->getConfig()->getKey("datasetPath").getValue(),app->getConfig()->getKey("namesPath").getValue(),
                                               app->getConfig()->getKey("inferencesPath").getValue(),app->getConfig()->getKey("namesPath").getValue(),ui->checkBox_evaluator_merge->isChecked(),
                                               ui->checkBox_evaluator_mix->isChecked(),ui->checkBox_evaluator_show_eval->isChecked());
    }
    catch (const std::string& msg){
        std::cout << "Exception detected: " << msg << std::endl;
    }
    catch (const std::exception &exc)
    {
        std::cout << "Exeption Detected: " << exc.what();
    }
    catch (...){
        std::cout << "Uknown exectip type" << std::endl;
    }
}

void MainWindow::handleDetectButton() {
    std::string outputPath = this->ui->textEdit_detectorOutPath->toPlainText().toStdString();
    bool useDepth = this->ui->checkBox_detector_useDepth->isChecked();
    bool singleEvaluation = this->ui->checkBox_detector_single->isChecked();


    try{
    SampleGeneratorHandler::Detector::process(ui->listView_detector_dataset, ui->listView_detector_names,ui->listView_detector_reader_imp,app->getConfig()->getKey("datasetPath").getValue(),
                                              ui->listView_detector_weights,ui->listView_detector_net_config,ui->listView_detector_imp,ui->listView_detector_names_inferencer,
                                              app->getConfig()->getKey("weightsPath").getValue(),app->getConfig()->getKey("netCfgPath").getValue(),outputPath,app->getConfig()->getKey("namesPath").getValue(),
                                              useDepth,singleEvaluation);
    }
    catch (const std::string& msg){
        std::cout << "Exception detected: " << msg << std::endl;
    }
    catch (const std::exception &exc)
    {
        std::cout << "Exeption Detected: " << exc.what();
    }
    catch (...){
        std::cout << "Uknown exectip type" << std::endl;
    }
}

void MainWindow::handleSelectOutputFolderButtonDetector() {
    QFileDialog *fd = new QFileDialog;
    QTreeView *tree = fd->findChild <QTreeView*>();
    tree->setRootIsDecorated(true);
    tree->setItemsExpandable(false);
    fd->setFileMode(QFileDialog::Directory);
    fd->setOption(QFileDialog::ShowDirsOnly);
    fd->setViewMode(QFileDialog::Detail);
    fd->setDirectory("/mnt/large/pentalo/deep/evaluations/");
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
    tree->setRootIsDecorated(true);
    tree->setItemsExpandable(false);
    fd->setFileMode(QFileDialog::AnyFile);
//    fd->setOption(QFileDialog::Show);
    fd->setViewMode(QFileDialog::Detail);
    fd->setDirectory("/mnt/large/Temporal/Series");
    int result = fd->exec();
    QString directory;
    if (result)
    {
        directory = fd->selectedFiles()[0];
        this->ui->textEdit_deployInputPath->setText(directory);
    }
}

void MainWindow::handleDeployerImpListViewChange(const QModelIndex& selected, const QModelIndex& deselected) {
    if (selected.data().toString() == "stream") {
        ui->deployer_param_groupBox->setEnabled(true);
        ui->groupBox_config_option->setEnabled(true);
        handleDeployerConfigFileOptionChange(ui->deployer_radioButton_manual->isChecked());
    } else if (selected.data().toString() == "camera") {
        ui->textEdit_deployInputPath->setEnabled(false);
        ui->pushButton_deploy_input->setEnabled(false);
        ui->deployer_param_groupBox->setEnabled(false);
        ui->groupBox_config_option->setEnabled(false);
    }
    else {
        ui->textEdit_deployInputPath->setEnabled(true);
        ui->pushButton_deploy_input->setEnabled(true);
        ui->deployer_param_groupBox->setEnabled(false);
        ui->groupBox_config_option->setEnabled(false);
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
        ui->groupBox_deployer_inferencer_params->setEnabled(true);
    } else {
        ui->groupBox_deployer_inferencer_params->setEnabled(false);
    }
}

void MainWindow::handleProcessDeploy() {
    std::string inputInfo = this->ui->textEdit_deployInputPath->toPlainText().toStdString();

    QGroupBox* deployer_params = this->ui->deployer_param_groupBox;
    QGroupBox* inferencer_params = this->ui->groupBox_deployer_inferencer_params;

    try{
        SampleGeneratorHandler::Deployer::process(ui->listView_deploy_input_imp,ui->listView_deploy_weights,
                                                  ui->listView_deploy_net_config,ui->listView_deploy_impl,ui->listView_deploy_names_inferencer, ui->pushButton_stop_deployer_process,
                                                  deployer_params, inferencer_params, app->getConfig()->getKey("weightsPath").getValue(),
                                                  app->getConfig()->getKey("netCfgPath").getValue(),app->getConfig()->getKey("namesPath").getValue(),inputInfo);
    }
    catch (const std::string& msg){
        std::cout << "Exception detected: " << msg << std::endl;
    }
    catch (const std::exception &exc)
    {
        std::cout << "Exception Detected: " << exc.what();
    }
    catch (...){
        std::cout << "Uknown exectip type" << std::endl;
    }
}
