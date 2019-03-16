//
// Created by frivas on 18/03/17.
//
#include <iostream>
#include <glog/logging.h>


void myFunction() {
    LOG(ERROR) << "fatal message";

}


int main(int argc, char **argv) {
    std::string logPath = "/home/frivas/devel/machine-learning/DeepLearningSuite/cmake-build-debug/test/GLOG/";
    google::InitGoogleLogging(argv[0]);
//    google::SetLogDestination(0, std::string(logPath + "info.log").c_str());
//    google::SetLogDestination(1, std::string(logPath + "warning.log").c_str());

    for (google::LogSeverity s = google::WARNING; s < google::NUM_SEVERITIES; s++)
        google::SetLogDestination(s, "");
    google::SetLogDestination(google::INFO, "log.log");
    FLAGS_alsologtostderr = 1;

    fLI::FLAGS_max_log_size = 1; //MB


    fLI::FLAGS_minloglevel=google::ERROR;

    LOG(INFO) << "Info message";
    LOG(WARNING) << "Warning message";
    LOG(ERROR) << "Error message";
    myFunction();
    LOG(INFO) << "no" << std::endl;
    int num_cookies = 11;
    LOG_IF(INFO, num_cookies > 10) << "Got lots of cookies";

    DLOG(INFO) << "Debug message";



    PCHECK(num_cookies == 4) << "Write failed!";


}
