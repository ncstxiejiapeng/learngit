#ifndef ANGLESOLVER_H
#define ANGLESOLVER_H
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

struct RM_ArmorDate{
    RotatedRect armor;
    float height;
    float cha;
    float tx = 0;
    float ty = 0;
    float tz = 0;
    float pitch = 0;
    float yaw = 0;
    bool IsSmall = true;
    bool IsHave = false;
};

class AngleSolver
{
public:
    AngleSolver() {}
    void GetArmorAngle( RM_ArmorDate & BestArmor);

private:

    float fBigArmorWidth=22.5;
    float fBigArmorHeight=5.5;
    float fSmallArmorWidth=13.5;
    float fSmallArmorHeight=5.5;

    void GetPoint2D( RM_ArmorDate & BestArmor,std::vector<cv::Point2f>&point2D);
    void GetPoint3D( RM_ArmorDate & BestArmor,std::vector<cv::Point3f>&point3D);
    void CountAngleXY(const std::vector<cv::Point2f>&point2D,const std::vector<cv::Point3f>&point3D, RM_ArmorDate & BestArmor);

    //相机内参
    cv::Mat caremaMatrix = (cv::Mat_<float>(3, 3) <<
           1410.227599422603, 0, 302.327719400284,
          0, 1409.640592939647, 261.1166093893049,
            0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat_<float>(1, 5) << -0.437010450595147, 0.9891484748013578, 0.0001966516072574012, -0.001516753072670458, -8.233173937143485);


};
#endif // ANGLESOLVER_H
