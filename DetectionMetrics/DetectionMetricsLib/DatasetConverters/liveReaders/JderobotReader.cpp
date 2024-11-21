//
// Created by frivas on 24/02/17.
//


#include "JderobotReader.h"

JderobotReader::JderobotReader(std::map<std::string, std::string>* deployer_params_map, const std::string& path):DatasetReader(true) {

    Config::Properties cfg;

    if (deployer_params_map == NULL) {
        LOG(WARNING) << "null" << '\n';
        int argc=2;
        char* argv[2];
        argv[0] = (char*)std::string("myFakeApp").c_str();
        argv[1] = (char*)path.c_str();
        cfg = Config::load(argc, argv);
    } else {
        LOG(INFO) << "not null" << '\n';
        std::map<std::string, std::string>::iterator iter;
        YAML::Node rootNode;  // starts out as null
        YAML::Node nodeConfig;

        for (iter = deployer_params_map->begin(); iter != deployer_params_map->end(); iter++) {
            LOG(INFO) << iter->first << " " << iter->second << '\n';
            nodeConfig[iter->first.c_str()] = iter->second.c_str();
            LOG(INFO) << "here" << '\n';
        }

        rootNode["Camera"] = nodeConfig;

        cfg = Config::Properties(rootNode);

        LOG(INFO) << "done" << '\n';

    }


    try{

        this->jdrc = new Comm::Communicator(cfg);

        this->camera = Comm::getCameraClient(jdrc, "Camera");


    } catch (const std::exception& ex) {
        LOG(ERROR) << ex.what();
    }


  /*  Ice::CommunicatorPtr ic;

    //todo hack
    int argc=2;
    char* argv[2];
    argv[0] = (char*)std::string("myFakeApp").c_str();
    argv[1] = (char*)IceConfigFile.c_str();


    ic = EasyIce::initialize(argc,argv);
    Ice::ObjectPrx base = ic->propertyToProxy("Cameraview.Camera.Proxy");
    Ice::PropertiesPtr prop = ic->getProperties();

    if (0==base)
        throw "Could not create proxy";


    this->camera = jderobot::CameraClientPtr (new jderobot::cameraClient(ic,"Cameraview.Camera."));

    if (! this->camera){
        throw "Invalid proxy";
    }
    this->camera->start();
*/
}

bool JderobotReader::getNextSample(Sample &sample) {


    JdeRobotTypes::Image myImage = this->camera->getImage();
    cv::Mat image = myImage.data;


    if (!image.empty()){
        sample.setSampleID(std::to_string(++this->sample_count));
	cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        sample.setColorImage(image);
        //sample.setDepthImage(image);
    }
    //LOG(INFO) << "Fetching" << '\n';

}
