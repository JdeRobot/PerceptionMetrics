//
// Created by frivas on 2/02/17.
//

#include "ClassTypeOwn.h"



ClassTypeOwn::ClassTypeOwn(const std::string& classID) {
    fillStringClassesVector();
    this->classID = std::string(classID);
}

ClassTypeOwn::ClassTypeOwn(int id){
    fillStringClassesVector();
    this->classID=this->classes[id];
}

void ClassTypeOwn::fillStringClassesVector() {
    classes.push_back("aeroplane");
    classes.push_back("apple");
    classes.push_back("backpack");
    classes.push_back("banana");
    classes.push_back("baseball bat");
    classes.push_back("baseball glove");
    classes.push_back("bear");
    classes.push_back("bed");
    classes.push_back("bench");
    classes.push_back("bicycle");
    classes.push_back("bird");
    classes.push_back("boat");
    classes.push_back("book");
    classes.push_back("bottle");
    classes.push_back("bowl");
    classes.push_back("broccoli");
    classes.push_back("bus");
    classes.push_back("cake");
    classes.push_back("car");
    classes.push_back("carrot");
    classes.push_back("cat");
    classes.push_back("cell phone");
    classes.push_back("chair");
    classes.push_back("clock");
    classes.push_back("cow");
    classes.push_back("cup");
    classes.push_back("diningtable");
    classes.push_back("dog");
    classes.push_back("donut");
    classes.push_back("elephant");
    classes.push_back("fire hydrant");
    classes.push_back("fork");
    classes.push_back("frisbee");
    classes.push_back("giraffe");
    classes.push_back("hair drier");
    classes.push_back("handbag");
    classes.push_back("horse");
    classes.push_back("hot dog");
    classes.push_back("keyboard");
    classes.push_back("kite");
    classes.push_back("knife");
    classes.push_back("laptop");
    classes.push_back("microwave");
    classes.push_back("motorbike");
    classes.push_back("mouse");
    classes.push_back("orange");
    classes.push_back("oven");
    classes.push_back("parking meter");
    classes.push_back("person");
    classes.push_back("pizza");
    classes.push_back("pottedplant");
    classes.push_back("refrigerator");
    classes.push_back("remote");
    classes.push_back("sandwich");
    classes.push_back("scissors");
    classes.push_back("sheep");
    classes.push_back("sink");
    classes.push_back("skateboard");
    classes.push_back("skis");
    classes.push_back("snowboard");
    classes.push_back("sofa");
    classes.push_back("spoon");
    classes.push_back("sports ball");
    classes.push_back("stop sign");
    classes.push_back("suitcase");
    classes.push_back("surfboard");
    classes.push_back("teddy bear");
    classes.push_back("tennis racket");
    classes.push_back("tie");
    classes.push_back("toaster");
    classes.push_back("toilet");
    classes.push_back("toothbrush");
    classes.push_back("traffic light");
    classes.push_back("train");
    classes.push_back("truck");
    classes.push_back("tvmonitor");
    classes.push_back("umbrella");
    classes.push_back("vase");
    classes.push_back("wine glass");
    classes.push_back("zebra");
    classes.push_back("person-falling");
    classes.push_back("person-fall");

}
