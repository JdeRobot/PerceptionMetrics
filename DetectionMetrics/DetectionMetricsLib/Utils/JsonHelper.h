//
// Created by frivas on 30/07/17.
//

#ifndef SAMPLERGENERATOR_JSONHELPER_H
#define SAMPLERGENERATOR_JSONHELPER_H


#include <vector>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

class JsonHelper{
public:
    template <typename T>
    static std::vector<T> as_vector(boost::property_tree::ptree const& pt, boost::property_tree::ptree::key_type const& key)
    {
        std::vector<T> r;
        for (auto& item : pt.get_child(key))
            r.push_back(item.second.get_value<T>());
        return r;
    }
};
#endif //SAMPLERGENERATOR_JSONHELPER_H
