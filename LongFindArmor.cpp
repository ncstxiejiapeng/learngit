#define DEBUG
#include<LongFindArmor.h>

ArmorColor color = BLUE;//!更改目标装甲颜色
int GrayValue =50;
int BinaryValue = 50;

LongFindArmor::LongFindArmor()
{
#ifdef DEBUG
    cv::namedWindow("BGR");
    cv::createTrackbar("gray","BGR",&GrayValue,200);
    cv::createTrackbar("binary","BGR",&BinaryValue,200);
#endif
}

bool LongFindArmor::IsHaveArmor(Mat & src){
    SrcImage = Mat::zeros(src.size(),src.type());
    src.copyTo(SrcImage);
    if(GetArmorData()){

        return false;
    }else{
        return false;
    }
    return false;
}

bool LongFindArmor::GetArmorData(){
    std::vector<cv::Mat>splits;
    cv::Mat element=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5,5));
    cv::Mat gray,binary;

    cv::split(SrcImage,splits);

    if(color == RED){
        cv::subtract(splits[2],splits[0],binary);
    }else{
        cv::subtract(splits[0],splits[2],binary);
    }
    cvtColor(SrcImage,gray,CV_RGB2GRAY);

#ifdef DEBUG
    threshold(gray,gray,GrayValue,255,CV_THRESH_BINARY);
    threshold(binary,binary,BinaryValue,255,CV_THRESH_BINARY);
#else
    threshold(binary,binary,0,255,CV_THRESH_BINARY | CV_THRESH_OTSU);
    threshold(gray,gray,0,255,CV_THRESH_BINARY | CV_THRESH_OTSU);
#endif

    dilate(binary,binary,element);
    imshow("灰度",gray);
    imshow("颜色分割",binary);

//    GetLeds(gray,gray_leds);
//    GetLeds(binary,binary_leds);

//    if(binary_leds.size()>=2&&gray_leds.size()>=2)
//        return true;
//    else
//        return false;
    return false;

}

void LongFindArmor::GetLeds(const cv::Mat &src, std::vector<cv::RotatedRect> &leds){
//    vector<std::vector<cv::Point>>contours;
    std::vector<std::vector<cv::Point>>contours;
    cv::findContours(src,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

//    for(auto contour:contours)
//    {
//        if(contour.size() < 15)continue;
//        cv::RotatedRect led=GetOrientation(contour);

//        float fLedWHRatio=float(led.size.width/led.size.height);
//        if((led.angle>45&&led.angle<135)&&(fLedWHRatio>1.2&&fLedWHRatio<8.0))
//        {
//            leds.push_back(led);
//        }
//        else continue;
//    }
//    if(leds.size()<2)return;
//    std::sort(leds.begin(),leds.end(),[](cv::RotatedRect &a1, cv::RotatedRect &a2) { return a1.center.x<a2.center.x; });
}

void LongFindArmor::GetTwoLedsGroup(std::vector<cv::RotatedRect> &gray_leds, vector<cv::RotatedRect> &binary_leds,std::vector<std::vector<cv::RotatedRect> > &TwoLedsGroup){
//    for(size_t i = gray_leds.size();i>=0;i--)
//    {
//        for(size_t j = binary_leds;j>=0;j--)
//        {
////            if()
//        }
//    }
}
