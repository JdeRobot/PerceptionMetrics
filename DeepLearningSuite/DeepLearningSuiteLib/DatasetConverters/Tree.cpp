#include "Tree.h"
#include <glog/logging.h>

Tree::Tree(std::string filename) {
    boost::property_tree::ptree tree;

    // Parse the XML into the property tree.
    boost::property_tree::read_xml(filename, tree);


    this->setClassName("root");


    BOOST_FOREACH(boost::property_tree::ptree::value_type &v, tree.get_child("mappings")) {

         boost::property_tree::ptree subtree = v.second;

         Tree* rootOfSubTree = new Tree();

         fillSubTree(subtree, rootOfSubTree);
         rootOfSubTree->parent = this;
         this->children.push_back(rootOfSubTree);

    }

}

Tree::Tree() {

}

void Tree::fillSubTree(boost::property_tree::ptree tree, Tree* root) {


    std::string name = tree.get<std::string>("name");

    root->setClassName(name);


    if( tree.count("children") != 0 )
    {
        BOOST_FOREACH(boost::property_tree::ptree::value_type &v, tree.get_child("children")) {
            // The data function is used to access the data stored in a node.
             boost::property_tree::ptree subtree = v.second;

             Tree* rootOfSubTree = new Tree();

             fillSubTree(subtree, rootOfSubTree);
             rootOfSubTree->parent = root;
             root->children.push_back(rootOfSubTree);
        }
    }


}

void Tree::insertChild(Tree* tree) {
     if (tree == NULL) {
          throw std::invalid_argument("Children Subtree Passed is NULL");
     }
     tree->setParent(this);
     this->children.push_back(tree);
}

void Tree::setParent(Tree* tree) {
     if (tree == NULL) {
          throw std::invalid_argument("Parent Tree Passed is NULL");
     }
     this->parent = tree;
}

void Tree::printChildren() {
     if (this->children.empty()) {
          LOG(INFO) << "This is a leaf node and has no children" << '\n';
          return;
     }

     std::vector<Tree*>::iterator it;

     for (it = this->children.begin(); it != this->children.end(); it++) {
          (*it)->printClassName();
     }
}

void Tree::printChildrenRecursive() {
    if (this->children.empty()) {
         //std::cout << "This is a leaf node and has no children" << '\n';
         return;
    }

    std::vector<Tree*>::iterator it;

    for (it = this->children.begin(); it != this->children.end(); it++) {
         (*it)->printClassName();
         (*it)->printChildrenRecursive();

    }

}

Tree* Tree::searchTree(std::string className) {
    Tree* result = NULL;
    if (this->className == className) {
        result = this;
        return result;
    }
    std::vector<Tree*>::iterator it;
    for (it = this->children.begin(); it != this->children.end(); it++ ) {
        result = (*it)->searchTree(className);
        if(result) {
            return result;
        }
    }
    return result;
}

std::vector<std::string> Tree::getImmediateSynonmys(std::string passedClassName) {

    std::vector<std::string> results;

    Tree* classSubTree = searchTree(passedClassName);

    if (classSubTree != NULL) {

        //classSubTree->printClassName();
        Tree* parent = classSubTree->parent;

        std::vector<Tree*>::iterator it;

        if (parent->className != "root")  {

            for (it = parent->children.begin(); it != parent->children.end(); it++) {
                if ((*it) != classSubTree) {            // passed tree
                    results.push_back((*it)->className);
                }
            }

        }


    }

    return results;

}

void Tree::printClassName() {
     LOG(INFO) << "Class Name is: " << this->className << '\n';
}

void Tree::setClassName(std::string passedClassName) {
     this->className = passedClassName;
}
