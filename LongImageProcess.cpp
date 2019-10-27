#include<LongImageProcess.h>
#include<AutExp.h>
#define USEING_VIDEO          //是否使用录像
#define LONGDEBUG              //调试
//#define MUNUALDEBUG              //调试
#define LONGIMSHOW           //显示
//#define AUTEXP                        //自动曝光
#define MUNUAL                      //手动曝光

volatile unsigned int proIdx = 0;                   //生产
volatile unsigned int consIdx = 0;                //消费
 int exp_vule=32;                                                    //曝光值
  int bright=18;                                             //标准亮度
  Mat LongSrcImage;

void LongImageProcess:: LongImageProducter(){
#ifdef USEING_VIDEO
     VideoCapture cap("/home/xiejiapeng/视频/7.avi");

#else
      myown::VideoCapture cap ("/dev/video0",3);
//      cap("/dev/video_external",3);
      cap.setExpousureTime(0, exp_vule);
      cap.setVideoFormat(640, 480, 1);
      cap.info();
      cap.startStream();
#ifdef AUTEXP
       AutExp(cap,(float)bright);

    #ifdef LONGDEBUG
              cv::namedWindow("bright");
       cv::createTrackbar("bright", "bright", &bright, 200);
    #endif
#else
#ifdef LONGDEBUG
      cv::namedWindow("exposure");
      cv::createTrackbar("exposure", "exposure", &exp_vule, 200);
#endif
//     cap.setVideoFormat(640, 480, 1);

#endif
#endif
         int cishu = 0;
      while(1){
          while (proIdx > consIdx) ;
#ifdef USEING_VIDEO
#else
#ifdef AUTEXP
#ifdef LONGDEBUG
              AutExp(cap,(float)bright);
#endif
#else
#ifdef LONGDEBUG
              cap.setExpousureTime(0, exp_vule);
#endif
#endif
#endif

#ifdef USEING_VIDEO
#else
         #ifdef AUTEXP
         if(cishu%30 == 0)
         {
             AutExp(cap,(float)bright);
             cout<<"帧数:"<<cishu<<endl;
         }
#endif
#endif
           cap>>LongSrcImage;
//           LongSrcImage = imread("/home/xiejiapeng/图片/2019-10-27 10-31-23屏幕截图.png");
            cishu++;
           proIdx++;
      }

 }

void LongImageProcess::ImageConsumer(){

         while(1)
         {
             while (proIdx < consIdx||proIdx == consIdx) ;
             consIdx++;

//             cap>>LongSrcImage;
//             imshow("处理后",LongSrcImage);
             LongFindArmor FinalArmorData;

             if(FinalArmorData.IsHaveArmor(LongSrcImage)){

             }


             waitKey(1);
         }
}
