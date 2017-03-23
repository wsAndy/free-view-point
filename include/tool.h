#ifndef TOOL_H
#define TOOL_H

#include "stdio.h"
#include "iostream"
#include "iomanip"
#include "fstream"
#include "sstream"
#include "vector"
#include "string"

#include "pcl/point_types.h"
#include "pcl/io/pcd_io.h"
#include "pcl/visualization/cloud_viewer.h"
#include "pcl/visualization/boost.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"

#include "Eigen/Dense"
#include "Eigen/Core"

using namespace cv;
using namespace std;
using namespace Eigen;


// this tool is designed to test the paper's algorithm.
// and it might not work well for video stream since there is no accelerate or multi-thread.
namespace fvv_tool
{

        struct ImageFrame{
		double mK[3][3];
		double mR[3][3];
		double mT[3];

                Matrix4d mP;
                // mP=[mK*mR mK*mT ]
                //    [ 0      1   ]

                // Z*x=mP*X

                Mat rgb;
                Mat dep;
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

    //show pointcloud
    void showPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cd_p);

    void showPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cd_p);

    void showParameter();

    // generate mP
    void generateP();

    // convert depth image's pixel value to an actual one.
    double getPixelActualDepth(unsigned char d);

    // convert depth to image pixel.
    double getPixelDepth(double dw);

    // rendering to novel viewpoint.
    void rendering(vector<int>& img_id, Matrix4d& targetP);

    // fusing two depth image in novel view point image plane
    void fusingDepth(Mat& left, Mat& right, Mat& target);

    //fusing two rgb image
    void fusingRgb(Mat& left_rgb, Mat& left_dep, vector<cv::Point2i>& left_vir_link_orig, Matrix<double,3,1>& left_T,
                   Mat& right_rgb, Mat& right_dep, vector<cv::Point2i>& right_vir_link_orig, Matrix<double,3,1>& right_T,
                   Mat& target, Matrix<double,3,1>& target_T);


    // my god, in this paper, when we project depth or rgb image to a virtual image plane, rgb and depth
    // is uncorrelation  !!!!

    // maybe I should not use pcl and define a struct directly...

    // infact , I think this two function should be operate in one function.
    // project from UV to XYZ
    void projFromUVToXYZ( Mat& rgb, Mat& dep, int img_index, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cd);

    void projFromUVToXYZ( Mat& dep, int img_index, pcl::PointCloud<pcl::PointXYZ>::Ptr cd);

    // project from XYZ to UV, since you need to project the pointcloud to a visual image plane
    void projFromXYZToUV( pcl::PointCloud<pcl::PointXYZRGB>::Ptr cd, Matrix4d &targetP, Mat& rgb, Mat& dep, std::vector<cv::Point>& vir_link_ori);

    void projFromXYZToUV( pcl::PointCloud<pcl::PointXYZ>::Ptr cd, Matrix4d &targetP, Mat& dep
                          , std::vector<cv::Point>& vir_link_ori);



    ImageFrame* cali;

private:
    int camera_num = 8;
    int MaxZ = 120;
    int MinZ = 44;

    int THRESHOLD = 5;

};


}
#endif // TOOL_H
