#ifndef LONGFINDARMOR_H
#define LONGFINDARMOR_H
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include<math.h>
#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;
using namespace std;

enum ArmorColor{
    RED,
    BLUE
};

struct RM_ArmorDate{
    RotatedRect armor;
    float height;
    float cha;
};

class LongFindArmor
{
public:
    bool IsHaveArmor(Mat & src);
    bool GetLedData();
    LongFindArmor();
    bool GetBestArmor();
    void GetLeds(const cv::Mat &gray, const cv::Mat &binary);
private:

    float small_max_ratio = 3.0;
    float small_min_ratio = 0.7;
    float big_max_ratio = 4.3;
    float big_min_ratio = 3.5;

    RotatedRect BestArmor;

    Mat SrcImage;
    void GetArmorDate(vector<Point> & leds,vector<RotatedRect> & ArmorDate);
    vector<cv::RotatedRect> ArmorDate;
     vector<RM_ArmorDate> _ArmorDate;
    void prediction(Mat & src);
};

#endif // LONGFINDARMOR_H
