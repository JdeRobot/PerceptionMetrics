//
// Created by frivas on 4/02/17.
//

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



SampleGenerationApp::SampleGenerationApp(int argc, char **argv):argc(argc),argv(argv) {

    if (parse_arguments(argc,argv,configFilePath) != SUCCESS){
        exit(1);
    }

    config = jderobotconfig::loader::load(configFilePath);
    //config=ConfigurationPtr( new Configuration(configFilePath));
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

void SampleGenerationApp::process() {
    if (verifyRequirements())
        (*this)();
}

bool SampleGenerationApp::verifyRequirements() {
    bool success=true;
    this->config.showConfig();
    std::string msg;
    for (auto it = this->requiredArguments.begin(), end =this->requiredArguments.end(); it != end; ++it){
        if (!this->config.keyExists(*it)){
            LOG(WARNING)<< "Key: " + *it + " is missing somewhere in the coniguration file";
            success=false;
        }

    }
    return success;
}

Config::Properties SampleGenerationApp::getConfig() {
    return config;
}
