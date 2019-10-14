//
// Created by frivas on 4/02/17.
//

// This is the main parent class of all the child like delpoyer,evaluator.
#include "SampleGenerationApp.h"
#include <glog/logging.h>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <iostream>

namespace
{
    const size_t ERROR_IN_COMMAND_LINE = 1;
    const size_t SUCCESS = 0;
    const size_t ERROR_UNHANDLED_EXCEPTION = 2;

} // namespace

// Constructor
SampleGenerationApp::SampleGenerationApp(int argc, char **argv):argc(argc),argv(argv) {
  // QApplication a(argc,argv);
  // this->a = new QApplication(argc,argv);
    // Check if command line arguments are passed or not , if not passed return error
    if (parse_arguments(argc,argv,configFilePath) != SUCCESS){
        exit(1);
    }
    // This loads the config file present at configFilePath which is passed
    config = jderobotconfig::loader::load(configFilePath);
    this->path= new std::string();
}

// Constructor which is called if a node itself is directly passed instead of
// configFilePath
SampleGenerationApp::SampleGenerationApp(YAML::Node node){
  config = jderobotconfig::loader::load(node);
}

// If a filepath is passed , it is loaded
SampleGenerationApp::SampleGenerationApp(std::string filepath, bool isPath){
  config = jderobotconfig::loader::load(filepath,true);
}

int SampleGenerationApp::parse_arguments(const int argc, char* argv[], std::string& configFile){
    for (google::LogSeverity s = google::WARNING; s < google::NUM_SEVERITIES; s++)
        google::SetLogDestination(s, "");
    google::SetLogDestination(google::INFO, "log.log");
    FLAGS_alsologtostderr = 1;
    fLI::FLAGS_max_log_size=10;

    try
    {
        /** Define and parse the program options
         */
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()
                ("help", "Print help messages")
                ("configFile,c", po::value<std::string>(&configFile)->required());
        po::variables_map vm;
        try
        {
            po::store(po::parse_command_line(argc, argv, desc),
                      vm); // can throw

            /** --help option
             */
            if ( vm.count("help")  )
            {
                LOG(INFO) << "Basic Command Line Parameter App" << std::endl
                          << desc << std::endl;
                return SUCCESS;
            }

            po::notify(vm); // throws on error, so do after help in case
            // there are any problems
        }
        catch(po::error& e)
        {
            LOG(ERROR) << "ERROR: " << e.what() << std::endl << std::endl;
            LOG(ERROR) << desc << std::endl;
            return ERROR_IN_COMMAND_LINE;
        }

        // application code here //

    }
    catch(std::exception& e)
    {
        LOG(ERROR) << "Unhandled Exception reached the top of main: "
                  << e.what() << ", application will now exit" << std::endl;
        return ERROR_UNHANDLED_EXCEPTION;

    }
    return SUCCESS;

}

// If all the requirements are satisfied this process further.
void SampleGenerationApp::process() {
    if (verifyRequirements())
        (*this)();
}

// Check if all the required Parameters like evalpath,classnamesfile, weigths etc are present or not
bool SampleGenerationApp::verifyRequirements() {
    bool success=true;
    this->config.showConfig();
    std::string msg;
    // We loop through the requiredArguments vector and check if every one of
    // them is present in the loaded configFile.
    for (auto it = this->requiredArguments.begin(), end =this->requiredArguments.end(); it != end; ++it){
        if (!this->config.keyExists(*it)){
            LOG(WARNING)<< "Key: " + *it + " is missing somewhere in the cofiguration file";
            // If certain Parameter is not present , a GUI is popped up to select
            // that parameter
            QApplication arm(this->argc,this->argv);
            pop_up win;
            win.SetPath(this->path);
            win.SetName(*it);
            win.show();
            arm.exec();
            // After selecting the property it is added to the config object
            this->config.SetProperty(*it,*(this->path));
            success=false;
            continue;
        }

    }
    // If not success , verify requirements again.
    if(!success)
      success=SampleGenerationApp::verifyRequirements();
    return success;
}

// Return the config parameter
Config::Properties SampleGenerationApp::getConfig() {
    return config;
}

// Destructor function
SampleGenerationApp::~SampleGenerationApp(){
}
