#include<AutExp.h>

//biaozhun值为适宜曝光下HSV亮度，bei值为需变化倍率
 float BiaoZhun = 70.0 ,bei = 1.0;
  int ExpValue=32;
//  int a = 30;

void AutExp( myown:: VideoCapture & cap,float biaozhun){
      int a = 30;
      Mat frame;
        cap>>frame;
        cvtColor(frame,frame,CV_RGB2HSV );
//              float biaozhun = 20.0 ,bei = 1.0;
       int SumV = 0;
         int size = frame.rows*frame.cols*3;
         for (int i = 2; i < size; i += 3)
         {
          SumV += frame.data[i];
      }
      int AvgV = 0;
      AvgV = SumV/(frame.rows*frame.cols);
   std:: cout<<"AvgV"<<AvgV<<std::endl;
   cout<<"BiaoZhun:"<<BiaoZhun<<"        biaozhun:"<<biaozhun<<endl;
    if(AvgV!=biaozhun)
    {
        BiaoZhun = biaozhun;
         while(a>=10||a<=-10){
             Mat frame;
               cap>>frame;
               cvtColor(frame,frame,CV_RGB2HSV );
//              float biaozhun = 20.0 ,bei = 1.0;
              int SumV = 0;
                int size = frame.rows*frame.cols*3;
                for (int i = 2; i < size; i += 3)
                {
                 SumV += frame.data[i];
             }
             int AvgV = 0;
             AvgV = SumV/(frame.rows*frame.cols);
                a = biaozhun - AvgV;
             if(a<0)
             {
                 ExpValue--;
                 cap.setExpousureTime(0, ExpValue);
             }else if (a>0){
                    if(ExpValue == 32){
                        ExpValue = 52;
                   }else {
                     ExpValue ++;
                 }
                    cap.setExpousureTime(0, ExpValue);

                }
             if(ExpValue == 60||ExpValue == 10)
                 break;
             cout<<ExpValue<<endl;
//        cout<<AvgV<<endl;
//                cvtColor(frame,frame,CV_HSV2BGR );
//                imshow("变化",frame);
          }
    }

}
