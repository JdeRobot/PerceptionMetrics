//
// Created by frivas on 19/02/17.
//

#include "Utils.h"

bool Utils::getListViewContent(const QListView *list, std::vector<std::string> &content, const std::string &prefix) {
    content.clear();

		if (list->model() == 0) {
			return false;
		}


    QModelIndexList selectedList =list->selectionModel()->selectedIndexes();
    for (auto it = selectedList.begin(), end = selectedList.end(); it != end; ++it){
        content.push_back(prefix + it->data().toString().toStdString());
    }

    return content.size() != 0;
}
