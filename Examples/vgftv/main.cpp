#include "tool.h"

using namespace std;
using namespace fvv_tool;
using namespace Eigen;



int main(int argc, char ** argv)
{
    Tool tool;
    tool.loadImageParameter("./../../dataset/MSR3DVideo-Breakdancers/calibParams-breakdancers.txt");

//    tool.showParameter();

    tool.generateP();

//    cout << tool.cali[0].mP <<endl;

    // now each tool.cali[] has mP, cali[4]

    /**
     * depth image operation, post-filtering
     *
     *  TODO
     *
     * **/

    Mat rgb(100,200,CV_8UC3,Scalar(0));
    Mat depth(100,200,CV_8UC1,Scalar(0));
    Matrix4d P;
    for(int i = 0;i <4;++i)
    {
        for(int j = 0 ; j < 4;++j)
        {
            P(i,j) = i*4+j;
        }
    }
    pcl::PointCloud<pcl::PointXYZRGB> cd;
    cd.width = 100;
    cd.height = 200;
    cd.resize(cd.width * cd.height);

    for(int i = 0 ; i < cd.height; ++i)
    {
        for(int j = 0; j < cd.width; ++j)
        {
            cd.points[i*cd.width+j].x = 1024 * rand()/ (RAND_MAX+1.0f);
            cd.points[i*cd.width+j].y = 1024 * rand()/ (RAND_MAX+1.0f);
            cd.points[i*cd.width+j].z = 1024 * rand()/ (RAND_MAX+1.0f);
        }
    }

//    tool.projFromXYZToUV(cd,P,rgb,depth);

//    rgb = Mat::ones(cd.height,cd.width,CV_8UC3);
//    depth = Mat::ones(cd.height,cd.width,CV_8UC1);

    for(int i = 0; i < 5;i++)
    {
        for(int j = 10; j < 20;++j)
        {
            cout << "out " << rgb.at<cv::Vec3b>(i,j)[0]<<endl;
        }
    }
    cout << "out" <<endl << depth.at<uchar>(1,1) <<endl;

    /**
     *  project the nearest image to viewpoint
     *
     *  TODO
     * */




    return 0;
}
