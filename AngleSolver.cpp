#include<AngleSolver.h>

 void AngleSolver::GetArmorAngle( RM_ArmorDate & BestArmor ){
     std::vector<cv::Point2f>point2D;
     std::vector<cv::Point3f>point3D;

     GetPoint2D(BestArmor,point2D);                                                                                                     //矩形转换为2d坐标
     GetPoint3D(BestArmor,point3D);                                                                                                     //矩形转换为3d坐标
     CountAngleXY(point2D,point3D,BestArmor);                                                                              //解决pnp问题
 }

 void AngleSolver::GetPoint2D( RM_ArmorDate & BestArmor,std::vector<cv::Point2f>&point2D){
     cv::Point2f p[4];
     BestArmor.armor.points(p);

     cv::Point2f lu,ld,ru,rd;        //right_down right_up left_up left_down
     //为坐标点排号
     std::sort(p,p + 4,[](const cv::Point2f &p1, const cv::Point2f &p2){ return p1.x < p2.x;});
     if(p[0].y < p[1].y)
     {
         lu = p[0];
         ld = p[1];
     }
     else {
         lu = p[1];
         ld = p[0];
     }
     if(p[2].y < p[3].y)
     {
         ru = p[2];
         rd = p[3];
     }
     else {
         ru = p[3];
         rd = p[2];
     }
     point2D.clear();///先清空再存入
     point2D.push_back(lu);
     point2D.push_back(ru);
     point2D.push_back(rd);
     point2D.push_back(ld);
 }

 //矩形转换为3d坐标                                                                                                                                                                                             3
 void AngleSolver::GetPoint3D( RM_ArmorDate & BestArmor,std::vector<cv::Point3f>&point3D)
 {
     float fHalfX=0;
     float fHalfY=0;
     if(BestArmor.IsSmall)                                  ///大装甲
     {
         fHalfX=fSmallArmorWidth/2.0;
         fHalfY=fSmallArmorHeight/2.0;
     }
     else{                                                                                                ///小装甲
         fHalfX=fBigArmorWidth/2.0;
         fHalfY=fBigArmorHeight/2.0;
     }
     point3D.push_back(cv::Point3f(-fHalfX,-fHalfY,0.0));
     point3D.push_back(cv::Point3f(fHalfX,-fHalfY,0.0));
     point3D.push_back(cv::Point3f(fHalfX,fHalfY,0.0));
     point3D.push_back(cv::Point3f(-fHalfX,fHalfY,0.0));
 }


 //pnp转换
 void AngleSolver::CountAngleXY(const std::vector<cv::Point2f>&point2D,const std::vector<cv::Point3f>&point3D, RM_ArmorDate & BestArmor)
 {
     cv::Mat rvecs=cv::Mat::zeros(3,1,CV_64FC1);
     cv::Mat tvecs=cv::Mat::zeros(3,1,CV_64FC1);

     cv::solvePnP(point3D,point2D,caremaMatrix,distCoeffs,rvecs,tvecs);

     double tx = tvecs.ptr<double>(0)[0];
     double ty = tvecs.ptr<double>(0)[1];
     double tz = tvecs.ptr<double>(0)[2];

     //换算
     tx -= 4.5 * (float)(tz/100);
     if(tz > 400)
     {
         tx -= 100;
     }
 //    ty -= 1.8 * (float)(tz/100);
     ty -= 3;
     float time = (float)(tz/1200);
     float h = 5 * time * time;
     ty -= h * 15;

     BestArmor.yaw = atan2(tx,tz)*180/CV_PI;
     BestArmor.pitch = atan2(ty,tz)*180/CV_PI;
     BestArmor.tx = tx;
     BestArmor.ty = ty;
     BestArmor.tz = tz;
 }
