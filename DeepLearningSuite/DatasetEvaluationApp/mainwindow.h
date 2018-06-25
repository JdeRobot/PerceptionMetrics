#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <Utils/SampleGenerationApp.h>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(SampleGenerationApp* app,QWidget *parent = 0);
    ~MainWindow();
    Ui::MainWindow *ui;


private:
    SampleGenerationApp* app;


private slots:
    void handleViewButton();
    void handleSelectOutputFolderButton();
    void handleSelectionNamesChanged();
    void setupTabsInformation();
    void handleConvertButton();
    void handleEvaluateButton();
    void handleDetectButton();
    void handleSelectOutputFolderButtonDetector();
    void handleSelectDeployInputSource();
    void handleProcessDeploy();
    void handleMappingCheckBoxChange();
    void handleDeployerImpListViewChange(const QModelIndex& selected, const QModelIndex& deselected);
    void handleDeployerConfigFileOptionChange(bool checked);
    void handleDeployerInferencerImpListViewChange(const QModelIndex& selected, const QModelIndex& deselected);
    void handleDetectorInferencerImpListViewChange(const QModelIndex& selected, const QModelIndex& deselected);
};

#endif // MAINWINDOW_H
