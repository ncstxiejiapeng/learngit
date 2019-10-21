#define DEBUG
#include<LongFindArmor.h>

ArmorColor color = BLUE;//!更改目标装甲颜色
int GrayValue =73;
int BinaryValue = 50;
vector<Point> leds;              //点形式存储灯条

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
    if(GetLedData()){

        GetArmorDate(leds,ArmorDate);
        if(GetBestArmor()){

        }
        imshow("绘图",SrcImage);
    }else{
        imshow("绘图",SrcImage);
        return false;
    }
    imshow("绘图",SrcImage);
    return false;
}

bool LongFindArmor::GetLedData(){
    std::vector<cv::Mat>splits;
    cv::Mat gray,binary;

    cv::split(SrcImage,splits);

    if(color == RED){
        cv::subtract(splits[2],splits[0],binary);
    }else{
        cv::subtract(splits[0],splits[2],binary);
    }
    cvtColor(SrcImage,gray,CV_RGB2GRAY);

#ifdef DEBUG
    threshold(splits[1],gray,GrayValue,255,CV_THRESH_BINARY);
    threshold(binary,binary,BinaryValue,255,CV_THRESH_BINARY);
#else
    threshold(binary,binary,0,255,CV_THRESH_BINARY | CV_THRESH_OTSU);
    threshold(gray,gray,0,255,CV_THRESH_BINARY | CV_THRESH_OTSU);
#endif

    cv::Mat element=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5,5));
    dilate(binary,binary,element);
    GetLeds(gray,binary);
    imshow("灰度",gray);
    imshow("颜色分割",binary);
//    waitKey(70);

    if(leds.size()>0){
        return true;
    }else {
        return false;
    }

//    GetLeds(gray,gray_leds);
//    GetLeds(binary,binary_leds);

//    if(binary_leds.size()>=2&&gray_leds.size()>=2)
//        return true;
//    else
//        return false;
    return false;

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
            if(pointPolygonTest(binary_contours[i], gray_contours[j][0], false) >= 0.0 ){

                cv::Mat data_contour=cv::Mat(binary_contours[i].size(),2,CV_64FC1);
                Point max,min;
                max = binary_contours[i][0];
                min = binary_contours[i][0];
                double max_y,min_y;
                max_y = min.y;
                min_y = min.y;
                for(int ii=0;ii<data_contour.rows;++ii)
                {
                    if(binary_contours[i][ii].y>max_y){
                        max_y = binary_contours[i][ii].y;
                        max = binary_contours[i][ii];
                    }
                    if(binary_contours[i][ii].y<min_y){
                        min_y =  binary_contours[i][ii].y;
                        min = binary_contours[i][ii];
                    }
                    data_contour.at<double>(ii,0)=binary_contours[i][ii].x;
                    data_contour.at<double>(ii,1)=binary_contours[i][ii].y;
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
                Point pos = Point(pca_analysis.mean.at<double>(0, 0),pca_analysis.mean.at<double>(0, 1));
                Point max_xiangliang;
                max_xiangliang = max - pos;
                float pos2 = (max_xiangliang.x*dir1.x+max_xiangliang.y*dir1.y)/sqrt(dir1.x*dir1.x+dir1.y*dir1.y);
                //计算出直线，在主要方向上绘制直线
                 line(SrcImage, pos, pos + Point(pos2*dir1.x,pos2*dir1.y) , CV_RGB(0, 0, 255),4);
                  line(SrcImage, pos + Point(pos2*dir1.x,pos2*dir1.y), pos - Point(pos2*dir1.x,pos2*dir1.y) , CV_RGB(0, 0, 255),4);


                   leds.push_back( pos - Point(pos2*dir1.x,pos2*dir1.y));
                    leds.push_back( pos + Point(pos2*dir1.x,pos2*dir1.y));

//                imshow("构型",SrcImage);
            }
        }
    }

}

void LongFindArmor::GetArmorDate(vector<Point> & leds,vector<RotatedRect> & ArmorDate){
    for(size_t i = 0;i<leds.size();i+=2){
        if(abs(leds[i].y-leds[i+1].y)<3)continue;
        for(size_t j = 0;j<leds.size();j+=2){
            if(abs(leds[j].y-leds[j+1].y)<3)continue;
            bool a = abs(leds[i].x-leds[j].x)>3;  //判断是否同为一个

            float angle2=atan2((leds[j].y-leds[j+1].y),(leds[j].x-leds[j+1].x))*180/CV_PI;
            float angle=atan2((leds[i].y-leds[i+1].y),(leds[i].x-leds[i+1].x))*180/CV_PI;


            bool b = atan2(abs(leds[j].y-leds[j+1].y),abs(leds[j].x-leds[j+1].x))*180/CV_PI>60;   //判断角度
            bool f = atan2(abs(leds[i].y-leds[i+1].y),abs(leds[i].x-leds[i+1].x))*180/CV_PI>60;   //判断角度
            bool g =fabs( (float)(angle - angle2))<50;
            float left = 0,right = 0;
            left =(leds[i].y-leds[i+1].y)*(leds[i].y-leds[i+1].y)+(leds[i].x-leds[i+1].x)*(leds[i].x-leds[i+1].x);
            right = (leds[j].y-leds[j+1].y)*(leds[j].y-leds[j+1].y)+(leds[j].x-leds[j+1].x)*(leds[j].x-leds[j+1].x);
            bool c = abs(left-right) < 5;
//            bool c = fabs((float)(fabs(leds[i].y-leds[i+1].y)) - (float)(fabs(leds[j].y-leds[j+1].y)))<3
//                                &&abs((float)(fabs((leds[i].x-leds[i+1].x))) - (float)(fabs(leds[j].x-leds[j+1].x)))<5;   //判断两者大小是否大概相等
            cout<<(left-right)<<endl;
            if(a&&c&&b&&f&&g){
                bool d = fabs(leds[i].x-leds[j].x)/fabs(leds[i].y-leds[i+1].y)>small_min_ratio&&fabs(leds[i].x-leds[j].x)/fabs(leds[i].y-leds[i+1].y)<small_max_ratio;
                bool e = fabs(leds[i].x-leds[j].x)/fabs(leds[i].y-leds[i+1].y)>big_min_ratio&&fabs(leds[i].x-leds[j].x)/fabs(leds[i].y-leds[i+1].y)<big_max_ratio;
                if(d||e){
//                    float angle1=atan2((leds[i].y-leds[j].y),abs(leds[i].x-leds[j].x))*180/CV_PI;
                    float angle1;
                    if(leds[i].x<leds[j].x){
                         angle1=atan2(abs(leds[i].x-leds[j].x),-(leds[i].y-leds[j].y))*180/CV_PI;
                    }else{
                         angle1=atan2(abs(leds[i].x-leds[j].x),(leds[i].y-leds[j].y))*180/CV_PI;
                    }
//                    float angle1=atan2(abs(leds[i].x-leds[j].x),-(leds[i].y-leds[j].y))*180/CV_PI;
                    Point center = Point((leds[i+1].x+leds[i].x+leds[j+1].x+leds[j].x)/4,(leds[i+1].y+leds[i].y+leds[j+1].y+leds[j].y)/4);
                    angle1 =- (angle1 - 90);
//                    if(angle1>90){
//                        angle1 = (180-angle1 );
//                    }
                    int width = (abs (leds[i].x-leds[j].x)+abs (leds[i+1].x-leds[j+1].x))/2;
                    int hight = (abs (leds[i].y-leds[i+1].y)+abs(leds[j].y-leds[j+1].y))/2;
                    RotatedRect Armor(center,Size(width,hight),angle1);
//                    Point2f vertices[4];
//                    Armor.points(vertices);

////                        line(SrcImage, leds[i], leds[i+1], Scalar(0,255,0));
////                        line(SrcImage, leds[j], leds[j+1], Scalar(0,255,0));
////                        line(SrcImage, leds[i], leds[j], Scalar(0,255,0));
////                        line(SrcImage, leds[i+1], leds[j+1], Scalar(0,255,0));

//                    for (int i = 0; i < 4; i++)
//                        line(SrcImage, vertices[i], vertices[(i+1)%4], Scalar(255,255,0));
//                     imshow("绘画",SrcImage);

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
    if(ArmorDate.size()==0){
        return false;
    }
    for(;i<ArmorDate.size();i++){
        if(abs(ArmorDate[i].angle)<10){
            BestArmor = ArmorDate[i];
        }
    }

    float Area = BestArmor.size.area();
    float angle = BestArmor.angle;

    for(;i<ArmorDate.size();i++){
        if(ArmorDate[i].size.area()>Area&&abs(ArmorDate[i].angle)<8){
            if((ArmorDate[i].size.area()-Area)<5){
                if(abs(ArmorDate[i].angle)<angle){
                    BestArmor = ArmorDate[i];
                }
            }else{
                Area = ArmorDate[i].size.area();
                BestArmor = ArmorDate[i];
            }
        }
    }
    Point2f vertices[4];
    BestArmor.points(vertices);

    for (int i = 0; i < 4; i++)
        line(SrcImage, vertices[i], vertices[(i+1)%4], Scalar(255,255,0));
//    waitKey(70);
    return true;
}
