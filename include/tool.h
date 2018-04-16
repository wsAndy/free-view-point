#ifndef TOOL_H
#define TOOL_H

#include "stdio.h"
#include "iostream"
#include "iomanip"
#include "fstream"
#include "sstream"
#include "vector"
#include "string"

//#include "pcl/point_types.h"
//#include "pcl/io/pcd_io.h"
//#include "pcl/visualization/cloud_viewer.h"
//#include "pcl/visualization/boost.h"

#include "opencv2/opencv.hpp"
//#include "opencv2/core/eigen.hpp"

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Core"

using namespace cv;
using namespace std;
using namespace Eigen;


// this tool is designed to test the paper's algorithm.
// and it might not work well for video stream since there is no accelerate or multi-thread.
namespace fvv_tool
{
    struct point{
        int r,g,b;
        double x,y,z;
    };
    struct pointcloud{
      int height;
      int width;
      vector<point> pl;
    };

    struct ImageFrame{

        double mK[3][3];
        double mR[3][3];
        double mT[3];

        Matrix3d K;
        Matrix4d RT;
        Matrix4d mP;
        // mP=[mK*mR mK*mT ]
        //    [ 0      1   ]

        // Z*x=mP*X

        vector<Mat> rgb_vec;
        vector<Mat> dep_vec; // 该值是世界坐标系的深度
//        vector<Mat> xyt_vec;
        vector<pointcloud> pl_vec; // 在世界坐标系中的值
        vector<int> proj_src_id; // 对一个特定位置而言，投影了哪一些id的图像，对应在投影位置的rgb_vec与dep_vec,如果一次就有同一个camid的多个序列位置的投影....哎乱了

        // 下面这两个还没有使用
        vector< vector<Mat> > pro_rgb; // 第一层是不同的序列，对应rgb_vec & dep_vec
        vector< vector<Mat> > pro_dep; // 第二层对应投影在不同位置，如在0，1，2，3

        vector< Mat> vir_img;


    };

    class Tool
    {
    public:
        Tool();
        Tool(string dataset_name, int num);
        ~Tool();

        // operate *cali ;
        void loadImageParameter(char* file_name);

        // load one image
        void loadImage(string& campath, vector<int>& camID, int startIndex = 0, int endIndex = 1);


        void showParameter();

        // generate mP
        void generateP();

        // get cali
        ImageFrame* getCamFrame();

        // return K,RT,mP
        void getParam(Eigen::Matrix4d& mP);

        // convert depth image's pixel value to an actual one.
        double getPixelActualDepth(unsigned char d);

        // convert depth to image pixel.
        double getPixelDepth(double dw);

        // rendering to novel viewpoint.
        void rendering(ImageFrame& img_frame);

        // smooth depth image
        void smoothDepth(ImageFrame& img_frame, int k_size);

        void warpuv(Matrix4d& src_mp, Matrix4d& dst_mp, double src_u, double src_v, double* dst_u, double* dst_v, Mat& dep);

        //fusing two rgb image
        void fusingRgb(Mat& left_rgb, Mat& left_dep, Matrix4d& left_mp, Matrix<double,3,1>& left_T,
                       Mat& right_rgb, Mat& right_dep, Matrix4d& right_mp, Matrix<double,3,1>& right_T,
                       Mat& vir_rgb, Matrix4d& target_mp, Matrix<double,3,1>& target_T);

        void colorConsistency(Mat& left_img, Mat &right_img);

        // id: image id that project.
        // startId: project which image from sequence? from 0 to  rgb_vec.size()-1
        // endId:   project which image from sequence? from 0 to  rgb_vec.size()-1
        void projUVtoXYZ(int id ,int startInd, int endInd);
        void projXYZtoUV(int cam_id, int startInd, int endInd, ImageFrame& tar_img);
        void writePLY(string name, pointcloud& pl);


        float distance(int x, int y, int i, int j) {
            return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
        }

        double gaussian(float x, double sigma) {
            return exp(-(pow(x, 2))/(2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
        }

        void applyBilateralFilter(Mat source, Mat filteredImage, int x, int y, int diameter, double sigmaI, double sigmaS);


        ImageFrame* cali;


        // 下面三个是使用提供投影代码的测试
        //    tool.forwardwarp(3, 4);
        //    tool.forwardwarp(5, 4);
        void forwardwarp(int src_camid, int dst_camid);
        // warp from (u1,v1) to (u2,v2)
        void projUVZtoXY(Eigen::Matrix4d &mp, double u, double v, double z, double *x, double *y, int height);
        double projXYZtoUV(Eigen::Matrix4d &mp, double x, double y, double z, double *u, double *v, int height);

    private:
        int camera_num = 8;
        int MaxZ = 120;  //120
        int MinZ = 44;   // 44

        int THRESHOLD = 5;

    };


}
#endif // TOOL_H
