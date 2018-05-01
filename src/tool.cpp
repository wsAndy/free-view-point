#include "tool.h"

using namespace std;
using namespace fvv_tool;
using namespace cv;

Tool::Tool()
{
    cout << "Tool init." <<endl;
    cali = new ImageFrame[camera_num];
}

Tool::Tool(string dataset_name, int cam_num)
{
    if( dataset_name.compare("ballet") == 0)
    {
        MaxZ = 130;
        MinZ = 42;
    }else{
        MaxZ = 120;
        MinZ = 44;
    }

    camera_num = cam_num;

    cali = new ImageFrame[camera_num];
}

Tool::~Tool()
{
    delete []cali;
    cout << "Tool clean" <<endl;
}

void Tool::writePLY(string name, pointcloud& pl)
{
    ofstream of(name,ios::out);

    of << "ply" << endl;
    of << "format ascii 1.0" << endl;
    of << "element vertex " << pl.pl.size() << endl;
    of << "property double x" << endl;
    of << "property double y" << endl;
    of << "property double z" << endl;
    of << "property uint8 red" << endl;
    of << "property uint8 green" << endl;
    of << "property uint8 blue" << endl;
    of << "end_header" << endl;
    for(int i = 0; i < pl.pl.size(); ++i)
    {
        of << pl.pl[i].x << " "
           << pl.pl[i].y << " "
           << pl.pl[i].z << " "
           << pl.pl[i].r << " "
           << pl.pl[i].g << " "
           << pl.pl[i].b << endl;
    }

    of.close();



}


void Tool::projUVZtoXY(Eigen::Matrix4d &mp, double u, double v, double z, double *x, double *y, int height)
{
    double c0,c1,c2;
    v = height - v - 1;

    c0 = z*mp(0,2) + mp(0,3);
    c1 = z*mp(1,2) + mp(1,3);
    c2 = z*mp(2,2) + mp(2,3);


    *y = u*( c1*mp(2,0) - c2*mp(1,0) )
            + v * (c2*mp(0,0)-c0*mp(2,0))
            + c0*mp(1,0) - c1*mp(0,0);
    *y/= v*( mp(2,0)*mp(0,1)-mp(2,1)*mp(0,0) )
            +u*(mp(1,0)*mp(2,1)-mp(1,1)*mp(2,0))
            +mp(0,0)*mp(1,1) - mp(1,0)*mp(0,1);
    *x = (*y)*(mp(0,1)-u*mp(2,1)) + c0 - c2*u;
    *x/= mp(2,0)*u - mp(0,0);

}

void Tool::projUVtoXYZ(int id ,int startInd, int endInd)
{
    for(int i = startInd; i < endInd; ++i)
    {
        pointcloud pl_, pl_f,pl_g;
        Mat dep = cali[id].dep_vec[i];
        Mat rgb = cali[id].rgb_vec[i];

        Mat front = cali[id].frontground[i];
        Mat back = cali[id].background[i];

        int height = dep.rows;
        int width = dep.cols;
        pl_.height = height;
        pl_.width = width;

        Matrix4d mp = cali[id].mP;

        for(int v = 0; v < height; ++v)
        {
            for(int u = 0; u < width; ++u)
            {

                double c0, c1, c2;

                double z = getPixelActualDepth(dep.at<Vec3b>(v,u)[0]);

                v = height - v - 1;

                c0 = z*mp(0,2) + mp(0,3);
                c1 = z*mp(1,2) + mp(1,3);
                c2 = z*mp(2,2) + mp(2,3);

                double x,y;
                y = u*( c1*mp(2,0) - c2*mp(1,0) )
                        + v * (c2*mp(0,0)-c0*mp(2,0))
                        + c0*mp(1,0) - c1*mp(0,0);
                y/= v*( mp(2,0)*mp(0,1)-mp(2,1)*mp(0,0) )
                        +u*(mp(1,0)*mp(2,1)-mp(1,1)*mp(2,0))
                        +mp(0,0)*mp(1,1) - mp(1,0)*mp(0,1);
                x = y*(mp(0,1)-u*mp(2,1)) + c0 - c2*u;
                x/= mp(2,0)*u - mp(0,0);

                point pp;
                pp.x = x;
                pp.y = y;
                pp.z = z;
                pp.address = mainground;

                if(front.at<cv::Vec3b>(height - v - 1,u)[0] > 250 &&
                   front.at<cv::Vec3b>(height - v - 1,u)[1] < 5 &&
                   front.at<cv::Vec3b>(height - v - 1,u)[2] < 5)
                {
                    pp.address = frontground;
                }

                if(back.at<cv::Vec3b>(height - v - 1,u)[0] < 5 &&
                 back.at<cv::Vec3b>(height - v - 1,u)[1] < 5 &&
                 back.at<cv::Vec3b>(height - v - 1,u)[2] >250)
                {
                    pp.address = background;
                }

                pp.r = rgb.at<cv::Vec3b>(height -1- v, u)[2];
                pp.g = rgb.at<cv::Vec3b>(height -1- v, u)[1];
                pp.b = rgb.at<cv::Vec3b>(height -1- v, u)[0];

                pl_.pl.push_back(pp);

            }
        }
        cali[id].pl_vec.push_back(pl_);
    }


}




double Tool::projXYZtoUV(Eigen::Matrix4d &P, double x, double y, double z, double *u, double *v, int height)
{
    double w;

    *u = P(0,0)*x + P(0,1)*y + P(0,2)*z + P(0,3);
    *v = P(1,0)*x + P(1,1)*y + P(1,2)*z + P(1,3);
    w = P(2,0)*x + P(2,1)*y + P(2,2)*z + P(2,3);

    *u = *u/w;
    *v = *v/w;

    *v = height - *v - 1;

    return w;
}

// 这边这个投影并不是最终的，要注意这个问题其实挺多的，需要讲究
// 这个target_img只对应一个源图像的投影的结果，对于多个源投影的结果，要有多个target_img
void Tool::projXYZtoUV(int cam_id, int startInd, int endInd, ImageFrame& target_img)
{
    ImageFrame src_img = cali[cam_id];
    Eigen::Matrix4d P = target_img.mP;
    Eigen::Matrix4d RT = target_img.RT;
    if(P.cols() != 4 || P.rows() !=4)
    {
        cerr << " targetP is not a [4x4] matrix!" <<endl;
    }else{
        cout << "project XYZ to UV start" <<endl;
        for(int cam = startInd; cam < endInd; cam++)
        {
            pointcloud pl_ = src_img.pl_vec[cam];
            int height = pl_.height;
            int width = pl_.width;

            Mat rgb_tar = Mat::zeros(height, width, CV_8UC3);
            Mat dep_tar = Mat::zeros(height, width, CV_8UC3);


            for(int i = 0; i < height; ++i)
            {
                for(int j = 0; j < width; ++j)
                {
                    double x,y,z,u,v,w;
                    x = pl_.pl[i*width + j].x; // 这一些都是在世界坐标系中的值
                    y = pl_.pl[i*width + j].y;
                    z = pl_.pl[i*width + j].z;

                    // 在这边加上，如果遇到前后景边缘，则将dep以及rgb变黑色不直接投影


                    u = P(0,0)*x + P(0,1)*y + P(0,2)*z + P(0,3);
                    v = P(1,0)*x + P(1,1)*y + P(1,2)*z + P(1,3);
                    w = P(2,0)*x + P(2,1)*y + P(2,2)*z + P(2,3);

                    u = u/w;
                    v = v/w;

                    v = height - v - 1;
                    // 这种投影一定有裂缝
                    int row = round( v );
                    int col = round( u );

// for debug
//                    if( i == 250 && j == 300)
//                    {
//                        cout << "x = " << x << ", y= " << y << " ,z = " << z << endl;
//                        cout << u << "," << v << endl;
//                        cout << "row = " << row << ", col = " << col << endl;
//                    }

                    if(row >= height || col >= width || row < 0 || col < 0)
                    {
                        continue;
                    }
                    if( dep_tar.at<cv::Vec3b>(row,col)[0] == 0 || getPixelActualDepth( dep_tar.at<cv::Vec3b>(row,col)[0] ) >= w )
                    {
                        dep_tar.at<cv::Vec3b>(row, col)[0] = getPixelDepth(w);
                        dep_tar.at<cv::Vec3b>(row, col)[1] = getPixelDepth(w);
                        dep_tar.at<cv::Vec3b>(row, col)[2] = getPixelDepth(w);

                        rgb_tar.at<cv::Vec3b>(row, col )[0] = pl_.pl[i*width + j].b;
                        rgb_tar.at<cv::Vec3b>(row, col )[1] = pl_.pl[i*width + j].g;
                        rgb_tar.at<cv::Vec3b>(row, col )[2] = pl_.pl[i*width + j].r;
//                        if( i == 250 && j == 300 )
//                        {
//                            rgb_tar.at<cv::Vec3b>(row,col)[0] = 0;
//                            rgb_tar.at<cv::Vec3b>(row,col)[1] = 0;
//                            rgb_tar.at<cv::Vec3b>(row,col)[2] = 255;
//                            cout << "set! " << endl;
//                        }
                    }

                }
            }

            target_img.rgb_vec.push_back(rgb_tar);
            target_img.dep_vec.push_back(dep_tar);
            target_img.proj_src_id.push_back(cam_id);
        }

    }

}

void Tool::showParameter()
{
    for(int k = 0; k < camera_num; ++k)
    {
        cout << "-----------"<<endl;
        cout << "camera " << k <<endl;
        cout << "camera intrinsic = " << endl;
        cout << cali[k].K << endl;
        cout << "camera Extrinsic = " << endl;
        cout << cali[k].RT << endl;
        cout << "camera  mP = " << endl;
        cout << cali[k].mP << endl;

    }

}


/*
*   load camera calibration parameter, the parameter in the following format:
*
*  CameraNumber
*  Intrinsic matrix
*  barrel distortion (can be ignored, since its always 0, 0)
*  Extrinsic Matrix
*
*
*/
void Tool::loadImageParameter(char* file_name)
{
//    fstream f_read;
//    f_read.open(file_name, ios::in);
    FILE *f_read;
    f_read = fopen(file_name,"r");

    if(!f_read)
    {
        cerr << "ERROR! [" << file_name <<"] not exist."<<endl;
    }else{
        cerr << "load calibration parameter ..." <<endl;

        int camIdx;
        double tmp; // dummy
        for(int k = 0; k < camera_num; ++k)
        {
            fscanf(f_read,"%d",&camIdx);

            // camera intrinsics
            for (int i = 0; i < 3; ++i)
            {
                fscanf(f_read,"%lf\t%lf\t%lf",
                       &(cali[camIdx].mK[i][0]),
                       &(cali[camIdx].mK[i][1]),
                       &(cali[camIdx].mK[i][2]));
            }

            fscanf(f_read,"%lf",&tmp);
            fscanf(f_read,"%lf",&tmp);

            // camera rotation and transformation
            for (int i = 0; i < 3; ++i)
            {

                for(int j = 0; j < 3; ++j)
                {
                    fscanf(f_read,"%lf",
                           &(cali[camIdx].mR[i][j]));
                }

                fscanf(f_read,"%lf",
                       &(cali[camIdx].mT[i]));

            }

//            double t1,t2,t3;
//            t1 = cali[camIdx].mT[0];
//            t2 = cali[camIdx].mT[1];
//            t3 = cali[camIdx].mT[2];

//            cali[camIdx].mT[0] = -1*(cali[camIdx].mR[0][0]*t1 + cali[camIdx].mR[0][1]*t2 + cali[camIdx].mR[0][2]*t3);
//            cali[camIdx].mT[1] = -1*(cali[camIdx].mR[1][0]*t1 + cali[camIdx].mR[1][1]*t2 + cali[camIdx].mR[1][2]*t3);
//            cali[camIdx].mT[2] = -1*(cali[camIdx].mR[2][0]*t1 + cali[camIdx].mR[2][1]*t2 + cali[camIdx].mR[2][2]*t3);


        }

        fclose(f_read);



        cout << "load OK." <<endl;

    }

}


ImageFrame* Tool::getCamFrame()
{
     return cali;
}

void Tool::warpuv(Matrix4d& src_mp, Matrix4d& dst_mp, double src_u, double src_v, double* dst_u, double* dst_v, Mat& depth)
{
    double z;
    double dstu = 0.0, dstv = 0.0;
    z = getPixelActualDepth(depth.at<Vec3b>(src_v,src_u)[0]);
    projUVZtoXY(src_mp, src_u, src_v,z, &dstu, &dstv, depth.rows);
    projXYZtoUV(dst_mp,dstu,dstv,z, dst_u, dst_v, depth.rows);

}

void Tool::forwardwarp(int src_camid, int dst_camid)
{


    Mat srcimg = cali[src_camid].rgb_vec[0];
    Mat depimg = cali[src_camid].dep_vec[0];
    Matrix4d dst_mP = cali[dst_camid].mP;
    Matrix4d src_mP = cali[src_camid].mP;
    Mat rgbdst = Mat::zeros(srcimg.rows,srcimg.cols,CV_8UC3);
    Mat depdst = Mat::zeros(srcimg.rows,srcimg.cols,CV_8UC3);

    double x=0.0, y=0.0, z;
    double dstu = 0.0, dstv = 0.0;
    for(int i = 0; i < srcimg.rows; ++i)
    {
        for(int j = 0; j < srcimg.cols; ++j)
        {
            z = getPixelActualDepth(depimg.at<Vec3b>(i,j)[0]);
            projUVZtoXY(src_mP, (double)j, (double)i,z, &x, &y, srcimg.rows);
            projXYZtoUV(dst_mP,x,y,z,&dstu,&dstv, srcimg.rows);

            int dstU = round(dstu);
            int dstV = round(dstv);

            if(dstU >= 0 && dstU < srcimg.cols && dstV >= 0 && dstV < srcimg.rows)
            {

                if(depdst.at<Vec3b>(dstV,dstU)[0] == 0 || getPixelActualDepth( depdst.at<Vec3b>(dstV,dstU)[0] ) >= z )
                {
                    depdst.at<Vec3b>(dstV,dstU)[0] = round(getPixelDepth(z));
                    depdst.at<Vec3b>(dstV,dstU)[1] = round(getPixelDepth(z));
                    depdst.at<Vec3b>(dstV,dstU)[2] = round(getPixelDepth(z));

                    rgbdst.at<Vec3b>(dstV,dstU)[0] = srcimg.at<Vec3b>(i,j)[0];
                    rgbdst.at<Vec3b>(dstV,dstU)[1] = srcimg.at<Vec3b>(i,j)[1];
                    rgbdst.at<Vec3b>(dstV,dstU)[2] = srcimg.at<Vec3b>(i,j)[2];

                }
            }

        }
    }


//    imwrite("/Users/sheng/Desktop/dep"+to_string(src_camid)+"_d.png", depdst);
//    imwrite("/Users/sheng/Desktop/rgb"+to_string(src_camid)+"_d.jpg", rgbdst);


}


/**
 *   load one image( rgb and depth )
 *
 *
 */

void Tool::loadImage(string& campath, vector<int>& camID, int startIndex, int endIndex)
{
    stringstream ss;

    for(int i = 0; i < camID.size(); ++i)
    {
        //color-cam1-f000.jpg, depth-cam1-f000.png

        for(int imgIndex = startIndex; imgIndex < endIndex; ++imgIndex)
        {
            string color_path,dep_path;

            ss << campath << camID[i] << "/color-cam"<<camID[i]<<"-f" << setfill('0')<< setw(3) << imgIndex <<".jpg";
            ss >> color_path;
            ss.clear();

            ss << campath << camID[i] << "/depth-cam"<<camID[i]<<"-f" << setfill('0')<< setw(3) << imgIndex <<".png";
            ss >> dep_path;
            ss.clear();

            cali[camID[i]].rgb_vec.push_back( imread(color_path.c_str()) ); // here can only save one
            cali[camID[i]].dep_vec.push_back( imread(dep_path.c_str()) );   // here can only save one

            color_path.clear();
            dep_path.clear();
        }
    }
    cout << "load Image OK" <<endl;
}



/**
*   use Eigen to calculate P matrix
*
**/
void Tool::generateP()
{
    for(int k = 0; k < camera_num; ++k)
    {

        Matrix3d eg_mk, eg_mr;
        Vector3d eg_mt;
        for(int i =0; i<3; ++i)
        {
            for(int j = 0; j < 3;++j)
            {
                eg_mk(i,j) = cali[k].mK[i][j];
                eg_mr(i,j) = cali[k].mR[i][j];
            }
            eg_mt(i) = cali[k].mT[i];
        }
        cali[k].K = eg_mk;


        Matrix4d eg_P, rt;
        eg_P.block<3,3>(0,0) = eg_mk*eg_mr;
        eg_P.block<3,1>(0,3) = eg_mk*eg_mt;

        eg_P(3,0) = 0.0;
        eg_P(3,1) = 0.0;
        eg_P(3,2) = 0.0;
        eg_P(3,3) = 1.0;

        cali[k].mP = eg_P;

        rt.block<3,3>(0,0) = eg_mr;
        rt.block<3,1>(0,3) = eg_mt;
        rt(3,0) = 0.0;
        rt(3,1) = 0.0;
        rt(3,2) = 0.0;
        rt(3,3) = 1.0;
        cali[k].RT = rt;

        Vector3d pos = -1 * (rt.block<3,3>(0,0)).inverse() * rt.block<3,1>(0,3);
        cali[k].pos = pos;

    }

    cout << "generate each camera's P, OK" <<endl;

}


double Tool::getPixelActualDepth(unsigned char d)
{
    return 1.0/((d/255.0)*(1.0/MinZ - 1.0/MaxZ) + 1.0/MaxZ);
}

double Tool::getPixelDepth(double dw)
{
    return 255.0*(MaxZ*MinZ)*(1.0/dw-1.0/MaxZ)/(MaxZ-MinZ);
}

void Tool::colorConsistency(Mat& left_img, Mat &right_img)
{
    if( left_img.channels() != 3 )
    {
        return ;
    }
    Mat left_hsv, right_hsv;
    cvtColor(left_img,left_hsv,CV_BGR2YUV);
    cvtColor(right_img,right_hsv,CV_BGR2YUV);

    vector<Mat> left, right;
    split(left_hsv, left);
    split(right_hsv, right);


    for(int ch = 0; ch < 3; ++ch)
    {
        Mat left_sin = left.at(ch);
        Mat right_sin = right.at(ch);

        cv::Scalar l_m, l_stddev,r_m, r_stddev;

        cv::meanStdDev(left_sin, l_m, l_stddev);
        cv::meanStdDev(right_sin, r_m, r_stddev);

        double l_mean, l_std, r_mean, r_std, mean_, std_;
        l_mean = l_m[0];
        l_std = l_stddev[0];
        r_mean = r_m[0];
        r_std = r_stddev[0];

        mean_ = (l_mean+r_mean)/2.0;
        std_ = (l_std + r_std)/2.0;

        for(int i = 0; i < left_sin.rows; ++i)
        {
            for(int j = 0; j < left_sin.cols; ++j)
            {
                uchar val = left_sin.at<uchar>(i,j);
                uchar new_val = val;
                new_val = (uchar)( (val-l_mean)*std_/l_std + mean_ );
                left_sin.at<uchar>(i,j) = new_val;

                val = right_sin.at<uchar>(i,j);
                new_val = val;
                new_val = (uchar)( (val-r_mean)*std_/r_std + mean_ );
                right_sin.at<uchar>(i,j) = new_val;
            }
        }

        left.at(ch) = left_sin;
        right.at(ch) = right_sin;
    }

    Mat left_new, right_new;
    merge(left,left_new);
    merge(right,right_new);

    cvtColor(left_new, left_img, CV_YUV2BGR);
    cvtColor(right_new, right_img, CV_YUV2BGR);

}

void Tool::findNearestCamId(Matrix4d& rt, int num, int& m1_, int& m2_)
{
    Vector3d pos = -1 * (rt.block<3,3>(0,0)).inverse() * (rt.block<3,1>(0,3));

    double min_1 = 10000.0, min_2 = 10000.0;
    int min_id1 = 0, min_id2 = 0;

//    cout << "-----" <<endl;
    for(int i = 0; i < num; ++i)
    {
        Matrix4d rt_ = cali[i].RT;
        Vector3d pos_ = -1 * (rt_.block<3,3>(0,0)).inverse() * (rt_.block<3,1>(0,3));
        double ll = distance(pos,pos_);
//        cout << "i = " << i << ll << endl;
        if( min_1 > ll )
        {
            min_2 = min_1;
            min_1 = ll;
            min_id2 = min_id1;
            min_id1 = i;
        }else if( min_2 > ll )
        {
            min_2 = ll;
            min_id2 = i;
        }

//        cout << "min_id1 = " <<  min_id1 << ", min_id2 = " << min_id2 << endl;
//        cout << "min = " << min_1 << ", min2 = " << min_2 << endl;

    }
//    cout << "====" << endl;
    m1_ = min_id1;
    m2_ = min_id2;

}

void Tool::releaseImageFrame(ImageFrame& img)
{

    img.rgb_vec.clear();
    img.dep_vec.clear();
    for(int i = 0; i < img.pl_vec.size(); ++i)
    {
        img.pl_vec[i].pl.clear();
    }
    img.pl_vec.clear();
    img.proj_src_id.clear();
    img.pro_rgb.clear();
    img.pro_dep.clear();
    img.vir_img.clear();
    img.frontground.clear();
    img.background.clear();

}

void Tool::rendering(ImageFrame& img_frame, double d1, double d2)
{
    // use two image

    // 这边left一定是比right距离要小的一个，而且 proj_src_id的顺序在之前也定下来了
    Matrix<double,3,1> left_T;
    Matrix<double,3,1> right_T;
    Matrix<double,3,1> target_T;

    Matrix4d left_mp, right_mp, target_mp;

    left_T = (cali[img_frame.proj_src_id[0]].mP).block(0,3,3,1);
    right_T = (cali[img_frame.proj_src_id[1]].mP).block(0,3,3,1);
    target_T = img_frame.mP.block(0,3,3,1);

    left_mp = cali[img_frame.proj_src_id[0]].mP;
    right_mp =  cali[img_frame.proj_src_id[1]].mP;
    target_mp = img_frame.mP;

    Mat left_rgb = cali[img_frame.proj_src_id[0]].rgb_vec[0];
    Mat left_dep = img_frame.dep_vec[0];

    Mat right_rgb = cali[img_frame.proj_src_id[1]].rgb_vec[0];
    Mat right_dep = img_frame.dep_vec[1];

    Mat vir_rgb = Mat::zeros(left_rgb.rows, left_rgb.cols, CV_8UC3);

//    waitKey(0);

    // 下面颜色校正存在问题
//    colorConsistency(left_rgb, right_rgb);

    Mat left_front, left_back, right_front, right_back;

    left_front = cali[ img_frame.proj_src_id[0] ].frontground[0];
    left_back = cali[img_frame.proj_src_id[0]].background[0];

    right_front = cali[img_frame.proj_src_id[1]].frontground[0];
    right_back = cali[img_frame.proj_src_id[1]].background[0];

//    if(d1 > d2)// left图像距离大，则使用right为标准
//    {
//        fusingRgb(right_rgb,right_dep, right_front, right_back, right_mp, right_T,left_rgb,left_dep, left_front, left_back, left_mp, left_T,  vir_rgb, target_mp, target_T );
//    }else{

        fusingRgb(left_rgb,left_dep, left_front, left_back, left_mp, left_T, right_rgb,right_dep, right_front, right_back, right_mp, right_T, vir_rgb, target_mp, target_T );
//    }

    img_frame.vir_img.push_back(vir_rgb);


//    imwrite("/Users/sheng/Desktop/result.jpg",vir_rgb);

    cout << "fusing rgb over." <<endl;

}



void Tool::renderingTest(ImageFrame& img_frame, double d1, double d2)
{
    // use two image

    Mat left_rgb = img_frame.rgb_vec[0];
    Mat left_dep = img_frame.dep_vec[0];

    Mat right_rgb = img_frame.rgb_vec[1];
    Mat right_dep = img_frame.dep_vec[1];

    Mat vir_rgb = Mat::zeros(left_rgb.rows, left_rgb.cols, CV_8UC3);

    // 下面颜色校正存在问题
//    colorConsistency(left_rgb, right_rgb);

    Mat left_front, left_back, right_front, right_back;

    left_front = cali[ img_frame.proj_src_id[0] ].frontground[0];
    left_back = cali[img_frame.proj_src_id[0]].background[0];

    right_front = cali[img_frame.proj_src_id[1]].frontground[0];
    right_back = cali[img_frame.proj_src_id[1]].background[0];

    fusingRgb_forward(left_rgb,left_dep,  right_rgb, right_dep, vir_rgb );

    img_frame.vir_img.push_back(vir_rgb);


//    imwrite("/Users/sheng/Desktop/result.jpg",vir_rgb);

    cout << "fusing rgb over." <<endl;

}



/**
 *   smooth depth image, you can use a complex algorithm or a simple one like medianBlur.
 *
 */

void Tool::smoothDepth(ImageFrame& img_frame, int k_size)
{
    for(int i = 0; i < img_frame.dep_vec.size(); ++i)
    {
        Mat dep = img_frame.dep_vec[i];
        cv::medianBlur(dep,dep,k_size);
        img_frame.dep_vec[i] = dep;
        cout << "i = " << i << endl;
    }

    // could clear small holes
    for(int i = 0; i < img_frame.dep_vec.size(); ++i)
    {
        Mat dep = img_frame.dep_vec[i];
        Mat element = getStructuringElement(MORPH_RECT, Size(3,3));
        morphologyEx(dep,dep,MORPH_CLOSE, element);
        img_frame.dep_vec[i] = dep;
    }

// 下面这个实验结果好像没什么效果...
//    for(int i = 0; i < img_frame.dep_vec.size(); ++i)
//    {
//        Mat dep = img_frame.dep_vec[i];
//        Mat dep_new = Mat::zeros(dep.rows, dep.cols, CV_8UC3);
//        for(int u = 2; u < dep.rows -2; ++u)
//        {
//            for(int v = 2; v < dep.cols - 2; ++v)
//            {
//                applyBilateralFilter(dep, dep_new, u, v, 5, 5, 20);
//            }
//        }
//    }

}


void Tool::getFrontBackGround(int camid, int startIndex, int endIndex )
{

    Mat edge;
    Mat dep_g;
    for(int ind = startIndex; ind < endIndex; ++ind)
    {
        Mat dep = cali[camid].dep_vec[0];
        Mat rgb = cali[camid].rgb_vec[0];
        cvtColor(dep,dep_g,CV_BGR2GRAY);
        Canny(dep_g,edge,80,255);

        Mat frontground = Mat::zeros(edge.rows, edge.cols, CV_8UC3);
        Mat background = Mat::zeros(edge.rows, edge.cols, CV_8UC3);

        for(int i = 0; i < edge.rows; ++i)
        {
            for(int j = 1; j < edge.cols-1; ++j)
            {
                if( edge.at<uchar>(i,j) > 250 )
                {
                    if( getPixelActualDepth(dep.at<Vec3b>(i,j-1)[0]) > getPixelActualDepth(dep.at<Vec3b>(i,j+1)[0]) )
                    {
                        // j-1: background
                        background.at<Vec3b>(i,j-1)[0] = 0;
                        background.at<Vec3b>(i,j-1)[1] = 0;
                        background.at<Vec3b>(i,j-1)[2] = 255;

                        frontground.at<Vec3b>(i,j+1)[0] = 255;
                        frontground.at<Vec3b>(i,j+1)[1] = 0;
                        frontground.at<Vec3b>(i,j+1)[2] = 0;
                    }else{
                        // j+1: background
                        background.at<Vec3b>(i,j+1)[0] = 0;
                        background.at<Vec3b>(i,j+1)[1] = 0;
                        background.at<Vec3b>(i,j+1)[2] = 255;

                        frontground.at<Vec3b>(i,j-1)[0] = 255;
                        frontground.at<Vec3b>(i,j-1)[1] = 0;
                        frontground.at<Vec3b>(i,j-1)[2] = 0;
                    }

                    if(  getPixelActualDepth(dep.at<Vec3b>(i,j-1)[0]) + getPixelActualDepth(dep.at<Vec3b>(i,j+1)[0]) > 2*getPixelActualDepth(dep.at<Vec3b>(i,j)[0]) )
                    {
                        frontground.at<Vec3b>(i,j)[0] = 255;
                        frontground.at<Vec3b>(i,j)[1] = 0;
                        frontground.at<Vec3b>(i,j)[2] = 0;
                    }else{
                        background.at<Vec3b>(i,j)[0] = 0;
                        background.at<Vec3b>(i,j)[1] = 0;
                        background.at<Vec3b>(i,j)[2] = 255;
                    }

                }
            }
        }

        cali[camid].background.push_back(background);
        cali[camid].frontground.push_back(frontground);
    }

}


/**
 * generate rgb image in novel viewpoint image plane
 *
 * input: left and right rgb image
 * output: virtual imahe plane rgb image
 */

/*

void Tool::fusingRgb(Mat &left_rgb, Mat &left_dep, Matrix4d& left_mp, Matrix<double,3,1>& left_T,
                     Mat &right_rgb, Mat &right_dep, Matrix4d& right_mp, Matrix<double,3,1>& right_T,
                     Mat &target_rgb, Matrix4d& target_mp, Matrix<double,3,1>& target_T)
{
    Array3d t1 = target_T - left_T;
    Array3d t2 = target_T - right_T;

    double sub_L = sqrt(t1.square().sum());
    double sub_R = sqrt(t2.square().sum());

    double alpha = (sub_L)/(sub_L + sub_R);

    double left_u = 0.0, left_v = 0.0, right_u = 0.0, right_v = 0.0 ;

    cout << "alpha: " << alpha << endl;

    for(int i = 0; i < left_rgb.rows; ++i)
    {
        for(int j = 0; j < left_rgb.cols; ++j)
        {

            if( left_dep.at<Vec3b>(i,j)[0] > 0 && right_dep.at<Vec3b>(i,j)[0] > 0 )
            {
                // 这边计算反向投影的对应点

                warpuv(target_mp, left_mp, (double)j, (double)i, &left_u, &left_v, left_dep);
                warpuv(target_mp, right_mp, (double)j, (double)i, &right_u, &right_v, right_dep);

                if( round(left_u) < 0 || round(left_u) > left_dep.cols || round(left_v) < 0 || round(left_v) >= left_dep.rows )
                {
                    continue;
                }
                if( round(right_u) < 0 || round(right_u) > left_dep.cols || round(right_v) < 0 || round(right_v) >= left_dep.rows )
                {
                    continue;
                }

                for(int ch = 0; ch < left_rgb.channels(); ++ch)
                {
                    target_rgb.at<Vec3b>(i,j)[ch] = (1-alpha) * left_rgb.at<Vec3b>(round(left_v), round(left_u) )[ch] + alpha * right_rgb.at<Vec3b>(round(right_v), round(right_u))[ch];
                }
            }else if(left_dep.at<Vec3b>(i,j)[0] == 0 && right_dep.at<Vec3b>(i,j)[0] >0 )
            {
                warpuv(target_mp, right_mp, (double)j, (double)i, &right_u, &right_v, right_dep);
                if( round(right_u) < 0 || round(right_u) > left_dep.cols || round(right_v) < 0 || round(right_v) >= left_dep.rows )
                {
                    continue;
                }
                for(int ch = 0; ch < left_rgb.channels(); ++ch)
                {
                    target_rgb.at<Vec3b>(i,j)[ch] = right_rgb.at<Vec3b>(round(right_v), round(right_u))[ch];
                }
            }else if( left_dep.at<Vec3b>(i,j)[0] > 0 && right_dep.at<Vec3b>(i,j)[0] == 0 )
            {
                warpuv(target_mp, left_mp, (double)j, (double)i, &left_u, &left_v, left_dep);
                if( round(left_u) < 0 || round(left_u) > left_dep.cols || round(left_v) < 0 || round(left_v) >= left_dep.rows )
                {
                    continue;
                }
                for(int ch = 0; ch < left_rgb.channels(); ++ch)
                {
                    target_rgb.at<Vec3b>(i,j)[ch] = left_rgb.at<Vec3b>(round(left_v), round(left_u) )[ch];
                }
            }else{
                ;
            }

        }
    }


}
*/


void Tool::fusingRgb_forward(Mat& left_rgb, Mat& left_dep,
                             Mat& right_rgb,Mat& right_dep,
                             Mat& vir_rgb)
{
//    Mat target_dep = Mat::zeros(left_rgb.rows, left_rgb.cols, CV_8UC3);
//    Mat target_back = Mat::zeros(left_rgb.rows, left_rgb.cols, CV_8UC3);

    for(int i = 0; i < left_rgb.rows; ++i)
    {
        for(int j = 0; j < left_rgb.cols; ++j)
        {
            if( left_dep.at<Vec3b>(i,j)[0] != 0 &&  left_dep.at<Vec3b>(i,j)[1] != 0 &&  left_dep.at<Vec3b>(i,j)[2] != 0)
            {
                for(int ch = 0; ch < 3; ++ch)
                {
                    vir_rgb.at<Vec3b>(i,j)[ch] = left_rgb.at<Vec3b>(i,j)[ch];
                }
            }else if(right_dep.at<Vec3b>(i,j)[0] != 0 &&  right_dep.at<Vec3b>(i,j)[1] != 0 &&  right_dep.at<Vec3b>(i,j)[2] != 0)
            {
                for(int ch = 0; ch < 3; ++ch)
                {
                    vir_rgb.at<Vec3b>(i,j)[ch] = right_rgb.at<Vec3b>(i,j)[ch];
                }
            }
        }
    }

// 本来下面有边缘的深度修正啥啥啥的先不做


}


void Tool::fusingRgb(Mat& left_rgb, Mat& left_dep, Mat& left_front, Mat& left_back, Matrix4d& left_mp, Matrix<double,3,1>& left_T,
               Mat& right_rgb, Mat& right_dep,Mat& right_front, Mat& right_back, Matrix4d& right_mp, Matrix<double,3,1>& right_T,
               Mat& vir_rgb, Matrix4d& target_mp, Matrix<double,3,1>& target_T)
{
    Array3d t1 = target_T - left_T;
    Array3d t2 = target_T - right_T;

    double sub_L = sqrt(t1.square().sum());
    double sub_R = sqrt(t2.square().sum());

    double alpha = (sub_L)/(sub_L + sub_R);

    double left_u = 0.0, left_v = 0.0, right_u = 0.0, right_v = 0.0 ;

    cout << "alpha: " << alpha << endl;

    Mat target_dep = Mat::zeros(left_rgb.rows, left_rgb.cols, CV_8UC3);
    Mat target_back = Mat::zeros(left_rgb.rows, left_rgb.cols, CV_8UC3);

    for(int i = 0; i < left_rgb.rows; ++i)
    {
        for(int j = 0; j < left_rgb.cols; ++j)
        {

            if( left_dep.at<Vec3b>(i,j)[0] > 0)
            {
                // 这边计算反向投影的对应点

                warpuv(target_mp, left_mp, (double)j, (double)i, &left_u, &left_v, left_dep);

                if( round(left_u) < 0 || round(left_u) >= left_dep.cols || round(left_v) < 0 || round(left_v) >= left_dep.rows )
                {
                    continue;
                }

                for(int ch = 0; ch < left_rgb.channels(); ++ch)
                {
                    vir_rgb.at<Vec3b>(i,j)[ch] = left_rgb.at<Vec3b>(round(left_v), round(left_u) )[ch];
                    target_dep.at<Vec3b>(i,j)[ch] = left_dep.at<Vec3b>(i,j)[ch];
//                    if( left_front.at<Vec3b>(round(left_v), round(left_u))[0] > 250 )
//                    {
//                        target_dep.at<Vec3b>(i,j)[0] = 255;
//                        target_dep.at<Vec3b>(i,j)[1] = 0;
//                        target_dep.at<Vec3b>(i,j)[2] = 0;
//                    }
                    if( left_back.at<Vec3b>(round(left_v), round(left_u))[2] > 250 )
                    {
                        target_back.at<Vec3b>(i,j)[0] = 0;
                        target_back.at<Vec3b>(i,j)[1] = 0;
                        target_back.at<Vec3b>(i,j)[2] = 255;
                    }
                }


            }else if(right_dep.at<Vec3b>(i,j)[0] >0 )
            {
                warpuv(target_mp, right_mp, (double)j, (double)i, &right_u, &right_v, right_dep);
                if( round(right_u) < 0 || round(right_u) >= left_dep.cols || round(right_v) < 0 || round(right_v) >= left_dep.rows )
                {
                    continue;
                }
                for(int ch = 0; ch < left_rgb.channels(); ++ch)
                {
                    vir_rgb.at<Vec3b>(i,j)[ch] = right_rgb.at<Vec3b>(round(right_v), round(right_u))[ch];
                    target_dep.at<Vec3b>(i,j)[ch] = right_dep.at<Vec3b>(i,j)[ch];
//                    if( right_front.at<Vec3b>(round(right_v), round(right_u))[0] > 250 )
//                    {
//                        target_dep.at<Vec3b>(i,j)[0] = 255;
//                        target_dep.at<Vec3b>(i,j)[1] = 0;
//                        target_dep.at<Vec3b>(i,j)[2] = 0;
//                    }
                    if( right_back.at<Vec3b>(round(right_v), round(right_u))[2] > 250 )
                    {
                        target_back.at<Vec3b>(i,j)[0] = 0;
                        target_back.at<Vec3b>(i,j)[1] = 0;
                        target_back.at<Vec3b>(i,j)[2] = 255;
                    }

                }
            }else{
                ;
            }

        }
    }

//    imwrite("/Users/sheng/Desktop/result.png",vir_rgb);
//    imwrite("/Users/sheng/Desktop/target_dep.png",target_dep);
//    imwrite("/Users/sheng/Desktop/target_dep_back.png",target_back);


    // 下面开始，在target_black中余下的点坐标周围，对vir_rgb做中值滤波

    Mat bback ;
    cvtColor(target_back, bback, CV_BGR2GRAY);
    threshold(bback,bback,20,255,CV_THRESH_BINARY);
    Mat element = getStructuringElement(MORPH_RECT, Size(3,3));
    morphologyEx(bback,bback,MORPH_DILATE, element);

//    imwrite("/Users/sheng/Desktop/bback.png",bback);

    for(int i = 2; i < bback.rows-2; ++i)
    {
        for(int j = 2; j < bback.cols-2; ++j)
        {
            if( bback.at<uchar>(i,j) > 100 )
            {
                // 在(i-2,j-2)-(i+2,j+2)之间做中值
                for(int ch = 0; ch < vir_rgb.channels(); ++ch)
                {
                    vector<uchar> tmp_v;
                    for(int ii = i -2; ii < i + 3; ++ii)
                    {
                        for(int jj = j -2; jj < j + 3; ++jj)
                        {
                            tmp_v.push_back( vir_rgb.at<Vec3b>(ii,jj)[ch] );
                        }
                    }
                    std::sort(tmp_v.begin(), tmp_v.end());

                    vir_rgb.at<Vec3b>(i,j)[ch] = tmp_v[ tmp_v.size()/2 ];

                }
            }
        }
    }


//    imwrite("/Users/sheng/Desktop/result2.png",vir_rgb);




}


/**
 * generate rgb image in novel viewpoint image plane
 *
 * input: left and right rgb image
 * output: virtual imahe plane rgb image
 */

void Tool::fusingRgb(Mat &left_rgb, Mat &left_dep, Matrix4d& left_mp, Matrix<double,3,1>& left_T,
                     Mat &right_rgb, Mat &right_dep, Matrix4d& right_mp, Matrix<double,3,1>& right_T,
                     Mat &target_rgb, Matrix4d& target_mp, Matrix<double,3,1>& target_T)
{
    Array3d t1 = target_T - left_T;
    Array3d t2 = target_T - right_T;

    double sub_L = sqrt(t1.square().sum());
    double sub_R = sqrt(t2.square().sum());

    double alpha = (sub_L)/(sub_L + sub_R);

    double left_u = 0.0, left_v = 0.0, right_u = 0.0, right_v = 0.0 ;

    cout << "alpha: " << alpha << endl;

    Mat target_dep = Mat::zeros(left_rgb.rows, left_rgb.cols, CV_8UC3);

    for(int i = 0; i < left_rgb.rows; ++i)
    {
        for(int j = 0; j < left_rgb.cols; ++j)
        {

            if( left_dep.at<Vec3b>(i,j)[0] > 0)
            {
                // 这边计算反向投影的对应点

                warpuv(target_mp, left_mp, (double)j, (double)i, &left_u, &left_v, left_dep);

                if( round(left_u) < 0 || round(left_u) >= left_dep.cols || round(left_v) < 0 || round(left_v) >= left_dep.rows )
                {
                    continue;
                }

                for(int ch = 0; ch < left_rgb.channels(); ++ch)
                {
                    target_rgb.at<Vec3b>(i,j)[ch] = left_rgb.at<Vec3b>(round(left_v), round(left_u) )[ch];
                    target_dep.at<Vec3b>(i,j)[ch] = left_dep.at<Vec3b>(i,j)[ch];
                }


            }else if(right_dep.at<Vec3b>(i,j)[0] >0 )
            {
                warpuv(target_mp, right_mp, (double)j, (double)i, &right_u, &right_v, right_dep);
                if( round(right_u) < 0 || round(right_u) >= left_dep.cols || round(right_v) < 0 || round(right_v) >= left_dep.rows )
                {
                    continue;
                }
                for(int ch = 0; ch < left_rgb.channels(); ++ch)
                {
                    target_rgb.at<Vec3b>(i,j)[ch] = right_rgb.at<Vec3b>(round(right_v), round(right_u))[ch];
                    target_dep.at<Vec3b>(i,j)[ch] = right_dep.at<Vec3b>(i,j)[ch];
                }
            }else{
                ;
            }

        }
    }

    imwrite("/Users/sheng/Desktop/target_dep.png",target_dep);


}


void Tool::applyBilateralFilter(Mat source, Mat filteredImage, int x, int y, int diameter, double sigmaI, double sigmaS) {
    double iFiltered = 0;
    double wP = 0;
    int neighbor_x = 0;
    int neighbor_y = 0;
    int half = diameter / 2;

    for(int i = 0; i < diameter; i++) {
        for(int j = 0; j < diameter; j++) {
            neighbor_x = x - (half - i);
            neighbor_y = y - (half - j);
            double gi = gaussian(source.at<Vec3b>(neighbor_x, neighbor_y)[0] - source.at<Vec3b>(x, y)[0], sigmaI);
            double gs = gaussian(distance(x, y, neighbor_x, neighbor_y), sigmaS);
            double w = gi * gs;
            iFiltered = iFiltered + source.at<Vec3b>(neighbor_x, neighbor_y)[0] * w;
            wP = wP + w;
        }
    }
    iFiltered = iFiltered / wP;
    filteredImage.at<Vec3b>(x, y)[0] = round(iFiltered);
    filteredImage.at<Vec3b>(x, y)[1] = round(iFiltered);
    filteredImage.at<Vec3b>(x, y)[2] = round(iFiltered);
}
