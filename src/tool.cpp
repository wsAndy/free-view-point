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
        pointcloud pl_;
        Mat dep = cali[id].dep_vec[i];
        Mat rgb = cali[id].rgb_vec[i];

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
        }

        fclose(f_read);

        cout << "load OK." <<endl;

    }

}


ImageFrame* Tool::getCamFrame()
{
     return cali;
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

void Tool::rendering(ImageFrame& img_frame)
{
    // use two image

    Matrix<double,3,1> left_T;
    Matrix<double,3,1> right_T;
    Matrix<double,3,1> target_T;

    left_T = (cali[img_frame.proj_src_id[0]].mP).block(0,3,3,1);
    right_T = (cali[img_frame.proj_src_id[1]].mP).block(0,3,3,1);
    target_T = img_frame.mP.block(0,3,3,1);

    Mat left_rgb = img_frame.rgb_vec[0];
    Mat left_dep = img_frame.dep_vec[0];

    Mat right_rgb = img_frame.rgb_vec[1];
    Mat right_dep = img_frame.dep_vec[1];

    Mat vir_rgb = Mat::zeros(left_rgb.rows, left_rgb.cols, CV_8UC3);

    /**
     *  reproject the pixel on virtual image plane to left image and right image
     *  get its rgb values
     *
     */
    fusingRgb(left_rgb,left_dep, left_T, right_rgb,right_dep, right_T, vir_rgb, target_T );

//    addWeighted(vir_rgb,0.5,cali[4].rgb,0.5,0,vir_rgb);
//    imshow("target",cali[4].rgb);

//    resize(vir_rgb,vir_rgb,Size(int(vir_rgb.cols/2),int(vir_rgb.rows/2)));

//    imshow("vir_rgb",vir_rgb);
    imwrite("/Users/sheng/Desktop/result.jpg",vir_rgb);
//    waitKey(0);
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
    }

}

/**
 * generate rgb image in novel viewpoint image plane
 *
 * input: left and right rgb image
 * output: virtual imahe plane rgb image
 */

void Tool::fusingRgb(Mat &left_rgb, Mat &left_dep,  Matrix<double,3,1>& left_T,
                     Mat &right_rgb, Mat &right_dep, Matrix<double,3,1>& right_T,
                     Mat &target_rgb, Matrix<double,3,1>& target_T)
{
    Array3d t1 = target_T - left_T;
    Array3d t2 = target_T - right_T;

    double sub_L = sqrt(t1.square().sum());
    double sub_R = sqrt(t2.square().sum());

    double alpha = (sub_L)/(sub_L + sub_R);

    cout << "alpha: " << alpha << endl;

    for(int i = 0; i < left_rgb.rows; ++i)
    {
        for(int j = 0; j < left_rgb.cols; ++j)
        {
            if( left_dep.at<Vec3b>(i,j)[0] > 0 && right_dep.at<Vec3b>(i,j)[0] > 0 )
            {
                for(int ch = 0; ch < left_rgb.channels(); ++ch)
                {
                    target_rgb.at<Vec3b>(i,j)[ch] = (1-alpha) * left_rgb.at<Vec3b>(i,j)[ch] + alpha * right_rgb.at<Vec3b>(i,j)[ch];
                }
            }else if(left_dep.at<Vec3b>(i,j)[0] == 0 && right_dep.at<Vec3b>(i,j)[0] >0 )
            {
                for(int ch = 0; ch < left_rgb.channels(); ++ch)
                {
                    target_rgb.at<Vec3b>(i,j)[ch] = right_rgb.at<Vec3b>(i,j)[ch];
                }
            }else if( left_dep.at<Vec3b>(i,j)[0] > 0 && right_dep.at<Vec3b>(i,j)[0] == 0 )
            {
                for(int ch = 0; ch < left_rgb.channels(); ++ch)
                {
                    target_rgb.at<Vec3b>(i,j)[ch] = left_rgb.at<Vec3b>(i,j)[ch];
                }
            }else{
                ;
            }

        }
    }
}



/**
 *  here need to accelerate
 *
 *  project from UV (image coordinate ) to XYZ (3D space)
 *
 *
 *  Z_w * x = P * X
 *      UV       XYZ
 *
 *  input: rgb , dep
 *  output: cd_
 **/
// this input dep is Vec3b ! not uchar
//void Tool::projFromUVToXYZ(Mat &rgb, Mat &dep, int index, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cd_)
//{
//    cd_->width = rgb.cols;
//    cd_->height = rgb.rows;
//    cd_->points.resize(cd_->width * cd_->height);

//    for(int i = 0; i < cd_->height; ++i)
//    {
//        for(int j = 0; j < cd_->width; ++j)
//        {

//            double zc = getPixelActualDepth(dep.at<cv::Vec3b>(i,j)[0]); // actual depth
//            int u = j;
//            int v = i;

////            cout << "depth: " << zc <<endl;

//            Matrix4d p_in = (cali[index].mP).inverse();
//            Vector4d x_(zc*u,zc*v,zc,1);
//            Vector4d X_;
//            X_ = p_in*x_;

//            cd_->points[i*cd_->width+j].x = X_(0);
//            cd_->points[i*cd_->width+j].y = X_(1);
//            cd_->points[i*cd_->width+j].z = X_(2);

//            cd_->points[i*cd_->width+j].r = rgb.at<cv::Vec3b>(i,j)[2];
//            cd_->points[i*cd_->width+j].g = rgb.at<cv::Vec3b>(i,j)[1];
//            cd_->points[i*cd_->width+j].b = rgb.at<cv::Vec3b>(i,j)[0];

//        }

//    }

//}


/**
 *  here need to accelerate
 *
 *  project from UV (image coordinate ) to XYZ (3D space)
 *
 *  only project depth
 *
 **/

//void Tool::projFromUVToXYZ( Mat &dep, int index, pcl::PointCloud<pcl::PointXYZ>::Ptr cd_)
//{
//    cd_->width = dep.cols;
//    cd_->height = dep.rows;
//    cd_->points.resize(cd_->width * cd_->height);

//    for(int i = 0; i < cd_->height; ++i)
//    {
//        for(int j = 0; j < cd_->width; ++j)
//        {
//            double zc = getPixelActualDepth(dep.at<cv::Vec3b>(i,j)[0]); // actual depth
//            int u = j;
//            int v = i;

//            Matrix4d p_in = (cali[index].mP).inverse();
//            Vector4d x_(zc*u,zc*v,zc,1);
//            Vector4d X_;
//            X_ = p_in*x_;

//            cd_->points[i*cd_->width+j].x = X_(0);
//            cd_->points[i*cd_->width+j].y = X_(1);
//            cd_->points[i*cd_->width+j].z = X_(2);

//        }

//    }

//}

/**
 *  project from XYZ (3D world coordinate) to UV (image coordinate)
 *
 *  input: cd_
 *  output: rgb, dep
 */

//void Tool::projFromXYZToUV(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cd_,
//                           Eigen::Matrix4d & targetP,
//                           Mat &rgb, Mat &dep,
//                           std::vector<cv::Point>& vir_link_ori)
//{
//    // here you need to initial rgb and depth first since not all the pixel in these two image will be fixed.

//    if(targetP.cols()!=4 || targetP.rows()!=4)
//    {
//        cerr << " targetP is not a [4x4] matrix!" <<endl;
//    }else{
//        // initial rgb and depth
//        // TODO

//        cout << "XYZRGB project UV .." <<endl;

//        rgb = Mat::zeros(cd_->height,cd_->width,CV_8UC3);
//        dep = Mat::zeros(cd_->height,cd_->width,CV_8UC1);

//        for(int i = 0; i < cd_->height; ++i)
//        {
//            for(int j = 0 ; j < cd_->width; ++j)
//            {
//                Vector4d X_;
//                X_(0) = cd_->points[i*cd_->width+j].x;
//                X_(1) = cd_->points[i*cd_->width+j].y;
//                X_(2) = cd_->points[i*cd_->width+j].z; // actual depth
//                X_(3) = 1.0;

//                double zc = X_(2);
//                Vector4d x_;
//                x_ = targetP*X_;

//                if(zc < 0.2) // important in test.
//                {
//                    vir_link_ori.push_back(Point(-1,-1));
//                    continue;
//                }
//                if(x_(0) < 0 || x_(1) < 0)
//                {
//                    vir_link_ori.push_back(Point(-1,-1));
//                    continue;
//                }

//                rgb.at<cv::Vec3b>(int(x_(1)/zc),int(x_(0)/zc))[0] = cd_->points[i*cd_->width+j].b;
//                rgb.at<cv::Vec3b>(int(x_(1)/zc),int(x_(0)/zc))[1] = cd_->points[i*cd_->width+j].g;
//                rgb.at<cv::Vec3b>(int(x_(1)/zc),int(x_(0)/zc))[2] = cd_->points[i*cd_->width+j].r;

//                dep.at<uchar>(int(x_(1)/zc),int(x_(0)/zc)) = getPixelDepth(zc);
//            }
//        }
//        cout << "XYZRGB project to UV .. OK" <<endl;
//    }

//}



/**
 *  project from XYZ (3D world coordinate) to UV (image coordinate)
 *
 *  input: cd_
 *  output: dep
 *
 *   only depth image
 */

//void Tool::projFromXYZToUV(pcl::PointCloud<pcl::PointXYZ>::Ptr cd_,
//                           Eigen::Matrix4d & targetP,
//                           Mat &dep,
//                           std::vector<cv::Point>& vir_link_ori)
//{

//    // here you need to initial rgb and depth first since not all the pixel in these two image will be fixed.

//    if(targetP.cols()!=4 || targetP.rows()!=4)
//    {
//        cerr << " targetP is not a [4x4] matrix!" <<endl;
//    }else{
//        // initial depth
//        // TODO

//        // here, you should not clear it if it is empty.
//        if(!vir_link_ori.empty())
//        {
//            vir_link_ori.clear();
//        }

//        cout << "Start project XYZ to UV.." <<endl;
//        for(int i = 0; i < cd_->height; ++i)
//        {
//            for(int j = 0 ; j < cd_->width; ++j)
//            {
//                Vector4d X_;
//                X_(0) = cd_->points[i*cd_->width+j].x;
//                X_(1) = cd_->points[i*cd_->width+j].y;
//                X_(2) = cd_->points[i*cd_->width+j].z; // actual depth
//                X_(3) = 1.0;

//                double zc = X_(2);
//                Vector4d x_;
//                x_ = targetP*X_;

//                // these judge operate is every important !!!!
//                // since the program will not stop if you locate a wide-point.
//                // especially the third one.
//                if(zc < 0.2) // important in test.
//                {
//                    vir_link_ori.push_back(Point(-1,-1));
//                    continue;
//                }

//                if(x_(0) < 0 || x_(1) < 0)
//                {
//                    vir_link_ori.push_back(Point(-1,-1));
//                    continue;
//                }

//                if( int(x_(1)/zc) >= dep.rows || int(x_(0)/zc) >= dep.cols)
//                {
//                    vir_link_ori.push_back(Point(-1,-1));
//                    continue;
//                }

//                dep.at<uchar>(int(x_(1)/zc),int(x_(0)/zc)) = getPixelDepth(zc);

//                vir_link_ori.push_back(Point(int(x_(0)/zc),int(x_(1)/zc)));
//            }
//        }

//        cout << "XYZ project to UV .. OK" << endl;
//    }

//}
