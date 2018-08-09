#include "mainwindow.h"
#include <QApplication>
#include <Utils/SampleGenerationApp.h>
#include <QStyleFactory>

class MyApp:public SampleGenerationApp{
public:
    MyApp(int argc, char* argv[]):SampleGenerationApp(argc,argv){
        this->requiredArguments.push_back("datasetPath");
        this->requiredArguments.push_back("evaluationsPath");
        this->requiredArguments.push_back("weightsPath");
        this->requiredArguments.push_back("netCfgPath");
        this->requiredArguments.push_back("namesPath");
    };
    void operator()(){
        QApplication a(argc, argv);
        MainWindow w(this);
        w.show();


        a.exec();

    };
};




int main(int argc, char *argv[])
{
    MyApp myApp(argc, argv);
    myApp.process();
}
