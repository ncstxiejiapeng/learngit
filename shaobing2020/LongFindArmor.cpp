#define DEBUG
#include<LongFindArmor.h>
#include <opencv2/ml.hpp>
#include <fstream>

using namespace cv::ml;

int xuhao = 1500;
char filename[200];
float BeiLv = 1.0;
Mat ROI;
Rect ArmorRoi;
int max_width = 640;
int max_hight = 480;
Point2f last_left_on = Point2f(0,0);
RM_ArmorDate BestArmor;

//大端转小端
int reverseInt(int i);

ArmorColor color = RED;//!更改目标装甲颜色
int GrayValue =49;
int BinaryValue = 32;
vector<Point2f> leds;              //点形式存储灯条
int number_vuler = 20;
 Ptr<SVM> svm = SVM::load("/home/xiejiapeng/xie_jia_peng/shaobing2020_test/train/svm_test/svm_new/svm0.xml");

LongFindArmor::LongFindArmor()
{
#ifdef DEBUG
    cv::namedWindow("BGR");
    cv::createTrackbar("gray","BGR",&GrayValue,200);
    cv::createTrackbar("binary","BGR",&BinaryValue,200);
#endif
}

bool LongFindArmor::IsHaveArmor(Mat & src){
    max_width = src.size().width;
    max_hight = src.size().height;
    src.copyTo(SrcImage);

    if(BestArmor.IsHave){
        ROI = src(ArmorRoi);
    } else {
        SrcImage.copyTo(ROI);
    }

    imshow("ROI",ROI);
    if(GetLedData()){
        GetArmorDate(leds,ArmorDate);
        if(ArmorDate.size() == 0){
            //未识别到进行左上点初始化
            last_left_on = Point(0,0);

            BestArmor.IsHave = false;
            imshow("绘图",SrcImage);
            return false;
        }else{
            if(GetBestArmor()){
                imshow("绘图",SrcImage);
                BestArmor.IsHave = true;
                return true;
            }else{
                //未识别到进行左上点初始化
                last_left_on = Point(0,0);

                imshow("绘图",SrcImage);
                BestArmor.IsHave = false;
                return false;
            }
        }
    }else{
        //未识别到进行左上点初始化
        last_left_on = Point(0,0);
        imshow("绘图",SrcImage);
        BestArmor.IsHave = false;
        return false;
    }
}

bool LongFindArmor::GetLedData(){
    std::vector<cv::Mat>splits;
    cv::Mat gray,binary;

//    GaussianBlur(SrcImage,SrcImage,Size(3,3),0,0);
    cv::split(ROI,splits);

    if(color == RED){
        cv::subtract(splits[2],splits[0],binary);
    }else{
        cv::subtract(splits[0],splits[2],binary);//    if(BestArmor.IsHave&&BestArmor.armor.size.area()>0){
        //        Rect ArmorRoi = Rect(BestArmor.armor.center.x-BestArmor.armor.size.width/2,BestArmor.armor.center.y-BestArmor.armor.size.height/2,BestArmor.armor.size.width*5,BestArmor.armor.size.height*5);
        //        ROI = src(ArmorRoi);
        //        rectangle(SrcImage, ArmorRoi, CV_RGB(0,255,0), 2, LINE_8);
        //    } else {
        //        src.copyTo(ROI);
        //    }
    }
//    cvtColor(ROI,gray,CV_RGB2GRAY);

#ifdef DEBUG
    threshold(splits[1],gray,GrayValue,255,CV_THRESH_BINARY);
    threshold(binary,binary,BinaryValue,255,CV_THRESH_BINARY);
#else
    threshold(binary,binary,0,255,CV_THRESH_BINARY | CV_THRESH_OTSU);
    threshold(gray,gray,0,255,CV_THRESH_BINARY | CV_THRESH_OTSU);
#endif

    cv::Mat element=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
     cv::Mat gray_element=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    dilate(binary,binary,element);
    dilate(gray,gray,gray_element);

    GetLeds(gray,binary);
    imshow("灰度",gray);
    imshow("颜色分割",binary);

    if(leds.size()>0){
        return true;
    }else {
        return false;
    }

}
//PCA算法获得两点构成直线形式灯条
void LongFindArmor::GetLeds(const cv::Mat &gray, const cv::Mat &binary){
//    vector<std::vector<cv::Point>>contours;
    std::vector<std::vector<cv::Point>>gray_contours;
    std::vector<std::vector<cv::Point>>binary_contours;
    cv::findContours(gray,gray_contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    cv::findContours(binary,binary_contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

    leds.clear();
    for(size_t i = 0; i < binary_contours.size(); i++){
        double area = contourArea(binary_contours[i]);
        if (area < 10.0 || 1e5 < area) continue;
        for(size_t j = 0; j < gray_contours.size(); j++){
            double gray_area = contourArea(gray_contours[j]);
            if(gray_area < 5.0 || 1e5 < gray_area) continue;
            size_t t = 0;    //计算比率
            for(size_t ii = 0;ii<(gray_contours[j].size());ii++){
                 if(pointPolygonTest(binary_contours[i], gray_contours[j][ii], false) > 0.0||pointPolygonTest(binary_contours[i], gray_contours[j][ii], false) == 0.0 ){
                     t++;
                 }
            }
            bool point = (t>gray_contours[j].size()/5);

            if(point){
                cv::Mat data_contour=cv::Mat(gray_contours[j].size(),2,CV_64FC1);
                Point2f max,min;
                max = gray_contours[j][0];
                min = gray_contours[j][0];
                float max_y,min_y;
                max_y = min.y;
                min_y = min.y;
                for(int ii=0;ii<data_contour.rows;++ii)
                {
                    if(gray_contours[j][ii].y>max_y){
                        max_y = gray_contours[j][ii].y;
                        max = gray_contours[j][ii];
                    }
                    if(gray_contours[j][ii].y<min_y){
                        min_y =  gray_contours[j][ii].y;
                        min = gray_contours[j][ii];
                    }
                    data_contour.at<double>(ii,0)=gray_contours[j][ii].x;
                    data_contour.at<double>(ii,1)=gray_contours[j][ii].y;
                }
                cv::PCA pca_analysis(data_contour,cv::Mat(),CV_PCA_DATA_AS_ROW);
                //存储特征向量和特征值
//                vector<Point2d> eigen_vecs(2);
//                vector<double> eigen_val(2);
//                for (int i = 0; i < 2; ++i)
//                {
//                    eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),pca_analysis.eigenvectors.at<double>(i, 1));
//                    eigen_val[i] = pca_analysis.eigenvalues.at<double>(i,0);//注意，这个地方原代码写错了
//                }

                cv::Point2f dir1;
                dir1.x = static_cast<float>( pca_analysis.eigenvectors.at<double>(0, 0) );
                dir1.y = static_cast<float>( pca_analysis.eigenvectors.at<double>(0, 1) );

                //排除横向
                if(dir1.y<0.7)continue;

                dir1 = dir1 * (1 / sqrt((float)dir1.x*dir1.x + (float)dir1.y*dir1.y));
//                dir2 = dir2 * (1 / sqrt((float)dir2.x*dir2.x + (float)dir2.y*dir2.y));
                Point2f pos = Point2f(pca_analysis.mean.at<double>(0, 0),pca_analysis.mean.at<double>(0, 1));
                Point2f max_xiangliang;
                max_xiangliang = max - pos;
                float pos2 = (max_xiangliang.x*dir1.x+max_xiangliang.y*dir1.y)/sqrt(dir1.x*dir1.x+dir1.y*dir1.y);
                //计算出直线，在主要方向上绘制直线
//                 line(SrcImage, pos, pos + Point(pos2*dir1.x,pos2*dir1.y) , CV_RGB(0, 0, 255),4);
//                  line(SrcImage, pos + Point2f(pos2*dir1.x,pos2*dir1.y), pos - Point2f(pos2*dir1.x,pos2*dir1.y) , CV_RGB(0, 0, 255),4);


                   leds.push_back( pos - Point2f(pos2*dir1.x,pos2*dir1.y));
                    leds.push_back( pos + Point2f(pos2*dir1.x,pos2*dir1.y));

//                imshow("构型",SrcImage);
            }
        }
    }

}

void LongFindArmor::GetArmorDate(vector<Point2f> & leds,vector<RotatedRect> & ArmorDate){
    _ArmorDate.clear();
    for(size_t i = 0;i<leds.size();i+=2){
        if(abs(leds[i].y-leds[i+1].y)<3)continue;
        for(size_t j = 0;j<leds.size();j+=2){
            if(abs(leds[j].y-leds[j+1].y)<3)continue;
            bool a = abs(leds[i].x-leds[j].x)>3;  //判断是否同为一个

            float angle2=atan2(abs(leds[j].y-leds[j+1].y),(leds[j].x-leds[j+1].x))*180/CV_PI;
            float angle=atan2(abs(leds[i].y-leds[i+1].y),(leds[i].x-leds[i+1].x))*180/CV_PI;


            bool b = atan2(abs(leds[j].y-leds[j+1].y),abs(leds[j].x-leds[j+1].x))*180/CV_PI>50;   //判断角度,没问题
            bool f = atan2(abs(leds[i].y-leds[i+1].y),abs(leds[i].x-leds[i+1].x))*180/CV_PI>50;   //判断角度,没问题
            bool g =fabs( (float)(angle - angle2))<6;               //角度差，没问题
//            cout<<"角度差："<< fabs( (float)(angle - angle2))<<endl;
//            cout<<"角度："<< atan2(abs(leds[i].y-leds[i+1].y),abs(leds[i].x-leds[i+1].x))*180/CV_PI<<endl;
            float left = 0,right = 0;

            left =sqrt(double(leds[i].y-leds[i+1].y)*(leds[i].y-leds[i+1].y)+(leds[i].x-leds[i+1].x)*(leds[i].x-leds[i+1].x));
            right = sqrt(double(leds[j].y-leds[j+1].y)*(leds[j].y-leds[j+1].y)+(leds[j].x-leds[j+1].x)*(leds[j].x-leds[j+1].x));
            bool c = abs(left-right) < 10;             //距离差，没问题
//            cout<<"距离差："<< abs(left-right)<<endl;
            if(a&c&&b&&f&&g){
//                cout<<"进"<<endl;
                bool d = fabs(leds[i].x-leds[j].x)/fabs(leds[i].y-leds[i+1].y)>small_min_ratio&&fabs(leds[i].x-leds[j].x)/fabs(leds[i].y-leds[i+1].y)<small_max_ratio;
                bool e = fabs(leds[i].x-leds[j].x)/fabs(leds[i].y-leds[i+1].y)>small_max_ratio&&fabs(leds[i].x-leds[j].x)/fabs(leds[i].y-leds[i+1].y)<big_max_ratio;
//                cout<<"比例："<<fabs(leds[i].x-leds[j].x)/fabs(leds[i].y-leds[i+1].y)<<endl;
                if(d||e){
//                    cout<<"进比例："<<fabs(leds[i].x-leds[j].x)/fabs(leds[i].y-leds[i+1].y)<<endl;
//                     cout<<"进距离差："<< abs(left-right)<<endl;
//                     cout<<"进角度差："<< fabs( (float)(angle - angle2))<<endl;
//                    float angle1=atan2((leds[i].y-leds[j].y),abs(leds[i].x-leds[j].x))*180/CV_PI;
                    float angle1,angle_led1,angle_led2,angle2;
                    if(leds[i].x<leds[j].x){
                         angle1=atan2(fabs(leds[i].x-leds[j].x),-(leds[i].y-leds[j].y))*180/CV_PI;
                         angle2=atan2(fabs(leds[i+1].x-leds[j+1].x),-(leds[i+1].y-leds[j+1].y))*180/CV_PI;
                         angle_led1=atan2(fabs(leds[i].y-leds[i+1].y),(leds[i].x-leds[i+1].x))*180/CV_PI;
                         angle_led2=atan2(fabs(leds[j].y-leds[j+1].y),(leds[j].x-leds[j+1].x))*180/CV_PI;
                    }else{
                         angle1=atan2(fabs(leds[i].x-leds[j].x),(leds[i].y-leds[j].y))*180/CV_PI;
                         angle2=atan2(fabs(leds[i+1].x-leds[j+1].x),(leds[i+1].y-leds[j+1].y))*180/CV_PI;
                         angle_led2=atan2(fabs(leds[j].y-leds[j+1].y),-(leds[j].x-leds[j+1].x))*180/CV_PI;
                         angle_led1=atan2(fabs(leds[i].y-leds[i+1].y),-(leds[i].x-leds[i+1].x))*180/CV_PI;

                    }
//                    cout<<(left+right)/2<<endl;
//                    cout<<"平行角度差： "<<fabs((angle_led1+angle_led2)/2.0-(angle1+angle2)/2)<<endl;
                    //根据灯条长度处理角度限制范围
                    float xian_zhi = (left+right)/2.0/4.0;
                    if(xian_zhi<5){
                        xian_zhi = 5;
                    }
//                    cout<<"平行角度差限制："<<xian_zhi<<endl;

                    //依据灯条长度动态修改平行角度差
                    if(fabs((angle_led1+angle_led2)/2.0-(angle1+angle2)/2)>xian_zhi)
                        continue;
                    Point2f center = Point2f((leds[i+1].x+leds[i].x+leds[j+1].x+leds[j].x)/4,(leds[i+1].y+leds[i].y+leds[j+1].y+leds[j].y)/4);
                    angle1 =- (angle1 - 90);
                    angle2 =- (angle2 - 90);
//                    if(angle1>90){
//                        angle1 = (180-angle1 );
//                    }
                    int width = (fabs (leds[i].x-leds[j].x)+fabs (leds[i+1].x-leds[j+1].x))/2;
                    int hight = (fabs (leds[i].y-leds[i+1].y)+fabs(leds[j].y-leds[j+1].y))/2;
                    RotatedRect Armor(center,Size(width,hight),(angle1+angle2)/2);
//                    Point2f vertices[4];
//                    Armor.points(vertices);

////                        line(SrcImage, leds[i], leds[i+1], Scalar(0,255,0));
////                        line(SrcImage, leds[j], leds[j+1], Scalar(0,255,0));
////                        line(SrcImage, leds[i], leds[j], Scalar(0,255,0));
////                        line(SrcImage, leds[i+1], leds[j+1], Scalar(0,255,0));

//                    for (int i = 0; i < 4; i++)
//                        line(SrcImage, vertices[i], vertices[(i+1)%4], Scalar(255,255,100));
//                     imshow("绘画",SrcImage);
                    RM_ArmorDate _Armor;
                    _Armor.armor = Armor;
//                    _ArmorDate.clear();
//                    cout<<"灯条1高度："<<fabs (leds[i].y-leds[i+1].y)<<endl<<"灯条2高度："<<fabs(leds[j].y-leds[j+1].y)<<endl;
                    _Armor.height = (fabs (leds[i].y-leds[i+1].y)+fabs(leds[j].y-leds[j+1].y))-fabs(fabs (leds[i].y-leds[i+1].y)-fabs(leds[j].y-leds[j+1].y));
                    _Armor.cha = fabs(fabs (leds[i].y-leds[i+1].y)-fabs(leds[j].y-leds[j+1].y));
//                    cout<<"jin jin 进"<<"高度："<<_Armor.height <<endl;

                    _ArmorDate.push_back(_Armor);

                    ArmorDate.push_back(Armor);
                }
            }else {
                continue;
            }


        }
    }
//    imshow("绘画",SrcImage);
}

bool LongFindArmor::GetBestArmor(){

    size_t i = 0;
    //高度2.0
    if(_ArmorDate.size()==0){
        return false;
        last_left_on = Point2f(0,0);
    }
    for(;i<_ArmorDate.size();i++){
        if(abs(_ArmorDate[i].armor.angle)<45){
            BestArmor.armor = _ArmorDate[i].armor;
            i++;
            break;
        }

    }
    if(i==_ArmorDate.size()){
        cout<<"BestArmor.size.area()==0"<<endl;
        last_left_on = Point2f(0,0);
        return false;
    }
    int height = BestArmor.armor.size.height;

    for(;i<_ArmorDate.size();i++){
        if(_ArmorDate[i].height>height&&abs(_ArmorDate[i].armor.angle)<45){
            if((_ArmorDate[i].height-height)<10){
                if(abs(_ArmorDate[i].armor.angle)<abs(BestArmor.armor.angle)){
                    BestArmor.armor = _ArmorDate[i].armor;
                    height = BestArmor.armor.size.height;
                }
            }else{
                height =BestArmor.armor.size.height;
                BestArmor.armor = _ArmorDate[i].armor;
            }
        }
    }
    Point2f vertices[4];
    BestArmor.armor.points(vertices);

    Point2f left_on = Point2f(0,0);
    if(BestArmor.IsHave){
        left_on = last_left_on;
    }

    //下一帧ROI
    int x = 0,y = 0,long_x = 0,long_y =  0;
    if(BestArmor.armor.center.x-BestArmor.armor.size.width*5/2+last_left_on.x>0&&BestArmor.armor.center.x-BestArmor.armor.size.width*5/2+last_left_on.x<=max_width){
        x = BestArmor.armor.center.x-BestArmor.armor.size.width*5/2;
    }else{
        last_left_on.x = 0;
        x = 0;
    }

    if(BestArmor.armor.center.y-BestArmor.armor.size.height*5/2+last_left_on.y>0&&BestArmor.armor.center.y-BestArmor.armor.size.height*5/2+last_left_on.y<=max_hight){
        y = BestArmor.armor.center.y-BestArmor.armor.size.height*5/2;
    }else {
        last_left_on.y = 0;
        y = 0;
    }

    if((x +BestArmor.armor.size.width*5+last_left_on.x)<max_width ){
        long_x = BestArmor.armor.size.width*5;
    }else{
        long_x = max_width-last_left_on.x - x;
    }

    if((y +BestArmor.armor.size.height*5+last_left_on.y)<max_hight){
        long_y = BestArmor.armor.size.height*5;
    }else{
        long_y = max_hight-last_left_on.y - y;
    }


    ArmorRoi = Rect(x+last_left_on.x,y+last_left_on.y,long_x,long_y);






    //数字ROI
    Mat rot_mat = getRotationMatrix2D(BestArmor.armor.center+last_left_on, BestArmor.armor.angle, 1.0);//求旋转矩阵
    Mat rot_image;
    Size dst_sz(SrcImage.size());
    warpAffine(SrcImage, rot_image, rot_mat, dst_sz);//原图像旋转
    float left_x = 0,left_y = 0,xuanzhuan_height = 0;
    if(BestArmor.armor.center.x- (BestArmor.armor.size.width *2/5)+left_on.x<0){
        left_x = 0;
    }else{
        left_x = BestArmor.armor.center.x - (BestArmor.armor.size.width *2/5)+left_on.x;
    }

    if(BestArmor.armor.center.y  - (BestArmor.armor.size.width/2)+left_on.y<0){
        left_y = 0;
    }else{
        left_y = BestArmor.armor.center.y - (BestArmor.armor.size.width/2)+left_on.y;
    }

    if(left_y +BestArmor.armor.size.width>max_hight){
        xuanzhuan_height = max_hight - left_y ;
    }else{
        xuanzhuan_height = BestArmor.armor.size.width;
    }

    Mat result1 = rot_image(Rect(left_x, left_y, BestArmor.armor.size.width*4/5, xuanzhuan_height));//提取ROI
    resize(result1,result1,Size(60,50));
        imshow("result", result1);
        prediction(result1);

    for (int i = 0; i < 4; i++)
        line(SrcImage,vertices[i]+last_left_on , vertices[(i+1)%4]+last_left_on, Scalar(255,255,0));

        last_left_on = Point2f(x+last_left_on.x,y+last_left_on.y);

    rectangle(SrcImage, ArmorRoi, CV_RGB(0,255,0), 2, LINE_8);

//    RM_ArmorDate armor;
//    BestArmor = armor;
    return true;
}

void LongFindArmor::prediction(Mat & src){

     resize(src,src,Size(28,28));

     cvtColor(src,src,CV_RGB2GRAY);
     threshold(src,src,0,255,CV_THRESH_BINARY | CV_THRESH_OTSU);
     imshow("字符",src);
     cvtColor(src,src,CV_GRAY2RGB);
//     String ImageName = "/home/xiejiapeng/xie_jia_peng/shaobing2020_test/train/4/"+to_string(xuhao++)+".jpg";
//     imwrite(ImageName,src);
//     cout<<"保存成功！"<<endl;

     Mat p = src.reshape(1, 1);
     p.convertTo(p, CV_32FC1);

     FinalArmor = BestArmor;
    float response = svm->predict(p);
    cout<<"预测结果："<<response<<endl;

}

void LongFindArmor::clear(){
    RotatedRect ling;
    BestArmor.armor = ling;
    BestArmor.tx = 0;
    BestArmor.ty = 0;
    BestArmor.tz = 0;
    BestArmor.yaw = 0;
    BestArmor.pitch = 0;
    BestArmor.height = 0;
}

//大端转小端
int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

