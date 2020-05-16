//
// Created by frivas on 22/11/16.
//

#include "BoundingRectGuiMover.h"

BoundingRectGuiMover::BoundingRectGuiMover(const std::vector<cv::Point> &points):points(points) {

}

BoundingRectGuiMover::BoundingRectGuiMover(const cv::Rect_<double> &rectangle) {
    points.push_back(cv::Point(rectangle.x,rectangle.y));
    points.push_back(cv::Point(rectangle.x + rectangle.width,rectangle.y));
    points.push_back(cv::Point(rectangle.x + rectangle.width,rectangle.y + rectangle.height));
    points.push_back(cv::Point(rectangle.x,rectangle.y + rectangle.height));
}




std::vector<cv::Point> BoundingRectGuiMover::getPoints() {
    return points;
}

void BoundingRectGuiMover::move(const cv::Point &from, const cv::Point &to,const  MovementType& type) {
    if (type == LOCAL_MOVEMENT) {
        unsigned int idx1, idx2;

        getClosestsLinePoints(from, idx1, idx2);

        cv::Point &p1 = points[idx1];
        cv::Point &p2 = points[idx2];


        //    std::cout << "closest points: " << p1 << ", " << p2<< std::endl;
        cv::Point movement;
        if (isVertical(p1, p2)) {
            movement = cv::Point(to.x - from.x, 0);
        } else {
            movement = cv::Point(0, to.y - from.y);
        }

        //    std::cout << "Vertical: " << isVertical(p1,p2) << "  Movement: " << movement << std::endl;
        //    std::cout << "Previous : " << p1 << ", " << p2 << std::endl;
        p1 = p1 + movement;
        p2 = p2 + movement;
    }
    else if (type == GLOBAL_MOVEMENT){
        cv::Point movement= to - from;
        for (auto it = points.begin(), end= points.end(); it != end; ++it){
            cv::Point &point = *it;
            point= point + movement;
        }
    }


}

void BoundingRectGuiMover::getClosestsLinePoints(const cv::Point &from, unsigned int &p1, unsigned int &p2) {

    double minDistance=999999999;

//    std::cout << "number of points: " << points.size() << std::endl;

    for (unsigned int i=0; i < points.size(); i++){
        int j=i+1;
        if (j >= points.size())
            j=0;

        cv::Point midpoint = (points[i] + points[j])*0.5;
        double distance = cv::norm(midpoint-from);
//        std::cout << "distance:" << distance  << std::endl;
        if (distance < minDistance){
            minDistance = distance;
            p1 = i;
            p2 = j;
        }
    }
}

bool BoundingRectGuiMover::isVertical(const cv::Point &from, const cv::Point &to) {
    int xMovement = abs(from.x-to.x);
    int yMovement = abs(from.y-to.y);
//    std::cout << "Xmov: " << xMovement << ", ymov: " << yMovement << std::endl;
    return abs(from.x-to.x) <  abs(from.y-to.y);
}

cv::Rect_<double> BoundingRectGuiMover::getRect(const double scale) {
    return cv::Rect_<double>(points[0].x/scale, points[0].y/scale,(points[1].x - points[0].x)/scale,(points[2].y - points[1].y)/scale);
}
