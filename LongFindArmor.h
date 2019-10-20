#ifndef LONGFINDARMOR_H
#define LONGFINDARMOR_H
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;
using namespace std;

enum ArmorColor{
    RED,
    BLUE
};

class LongFindArmor
{
public:
    bool IsHaveArmor(Mat & src);
    bool GetArmorData();
    LongFindArmor();
private:
    Mat SrcImage;
    void GetLeds(const cv::Mat &src, std::vector<cv::RotatedRect> &leds);
    vector<cv::RotatedRect> binary_leds;
     vector<cv::RotatedRect> gray_leds;
     void GetTwoLedsGroup(std::vector<cv::RotatedRect> &gray_leds, vector<cv::RotatedRect> &binary_leds, std::vector<std::vector<cv::RotatedRect> > &TwoLedsGroup);
};

#endif // LONGFINDARMOR_H
