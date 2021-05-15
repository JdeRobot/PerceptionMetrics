//
// Created by frivas on 4/02/17.
//

#ifndef SAMPLERGENERATOR_KEY_H
#define SAMPLERGENERATOR_KEY_H

#include <string>
#include <vector>

struct Key {
    Key();
    explicit Key(const std::string& key);



    bool isVector();
    void addValue(const std::string& value);
    std::string getValue();
    std::string getKey();
    std::string getValue(int id);
    std::string getValueOrLast(int id);           //return last value if id overflows
    std::vector<std::string> getValues();
    int getValueAsInt();

    int getNValues();

private:
    std::string key;
    std::vector<std::string> values;
    bool valid;
};


#endif //SAMPLERGENERATOR_KEY_H
