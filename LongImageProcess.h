#ifndef LONGIMAGEPROCESS_H
#define LONGIMAGEPROCESS_H
#include <opencv2/core/core.hpp>
#include<VideoCapture.h>
#include<LongFindArmor.h>

using namespace cv;

//       电控发数的结构体初始化
struct SendArmorData{
    float pitch = 0;
    float yaw = 0;
    unsigned char HaveArmor = 0;
    unsigned char dis = 0;
    unsigned char shoot = 0;
};

class LongImageProcess
{
public:
    LongImageProcess() {}
    void LongImageProducter();
    void ImageConsumer();

private:
//    Mat LongSrcImage;           //长焦镜头输入图像

};

#endif // LONGIMAGEPROCESS_H
