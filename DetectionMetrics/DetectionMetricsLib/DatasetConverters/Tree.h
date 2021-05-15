#include <iostream>
#include <string>
#include <vector>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>

class Tree {

public:
    // Member Functions()
    Tree(std::string filename);
    Tree();
    void fillSubTree(boost::property_tree::ptree tree, Tree* root);
    void printClassName();
    void printChildren();
    void printChildrenRecursive();
    void setClassName(std::string passedClassName);
    void insertChild(Tree* child);
    void setParent(Tree* parent);
    Tree* searchTree(std::string className);
    std::vector<std::string> getImmediateSynonmys(std::string passedClassName);

private:
    Tree* parent;
    std::string className;
    std::vector<Tree*> children;

};
