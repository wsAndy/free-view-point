#ifndef TOOL_H
#define TOOL_H

#include "stdio.h"
#include "iostream"
#include "fstream"
#include "vector"
#include "string"
#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
#include "pcl/point_types.h"
#include "pcl/io/pcd_io.h"
#include "opencv2/core/eigen.hpp"

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
    Tool(string dataset_name);
    ~Tool();

    // operate *cali ;
    void loadImageParameter(char* file_name);

    void showParameter();

    // generate mP
    void generateP();

    // convert depth image's pixel value to an actual one.
    double getPixelActualDepth(unsigned char d);

    // convert depth to image pixel.
    double getPixelDepth(double dw);

    // rendering to novel viewpoint.
    void rendering(vector<int>& img_id, Matrix4d& targetP);

    // my god, in this paper, when we project depth or rgb image to a virtual image plane, rgb and depth
    // is uncorrelation  !!!!

    // maybe I should not use pcl and define a struct directly...

    // infact , I think this two function should be operate in one function.
    // project from UV to XYZ
    void projFromUVToXYZ( Mat& rgb, Mat& dep, int img_index, pcl::PointCloud<pcl::PointXYZRGB>& cd);

    void projFromUVToXYZ( Mat& dep, int img_index, pcl::PointCloud<pcl::PointXYZ>& cd);

    // project from XYZ to UV, since you need to project the pointcloud to a visual image plane
    void projFromXYZToUV( pcl::PointCloud<pcl::PointXYZRGB>& cd, Matrix4d &targetP, Mat& rgb, Mat& dep);

    void projFromXYZToUV( pcl::PointCloud<pcl::PointXYZ>& cd, Matrix4d &targetP, Mat& dep);



    ImageFrame* cali;

private:
    int camera_num = 8;
    int MaxZ = 120;
    int MinZ = 44;

    pcl::PointCloud<pcl::PointXYZ> cld_map; // this global pointcloud might not be used
                                            // since you optimite those fusing image in the virtual image plane.

};


}
#endif // TOOL_H
