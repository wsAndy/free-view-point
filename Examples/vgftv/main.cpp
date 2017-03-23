#include "tool.h"


using namespace std;
using namespace cv;
using namespace fvv_tool;
using namespace Eigen;


void test(Tool& );

int main(int argc, char ** argv)
{
    int camera_num = 8;
    Tool tool; // Tool tool("ballet",8);

    char* path_parameter = "./../../dataset/MSR3DVideo-Breakdancers/calibParams-breakdancers.txt";
    tool.loadImageParameter(path_parameter);

//    tool.showParameter();

    tool.generateP(); // after loadImageParameter, generate each image's P according to reference camera 4

    // now each tool.cali[] has mP, cali[4]


    //load camera's image into tool.cali
    string path = "./../../dataset/MSR3DVideo-Breakdancers/cam";

    vector<int> camID;

    camID.push_back(0);
    camID.push_back(7);

    tool.loadImage(path,camID);// image's startIndex = 0, endIndex = 1 defaultly

    /**
     *  from UV to XYZ , show me pcl_viewer
     *
     *  TODO
     *
     * **/

    pcl::PointCloud<pcl::PointXYZ>::Ptr cd_p( new pcl::PointCloud<pcl::PointXYZ>);

    tool.projFromUVToXYZ(tool.cali[camID[0]].dep,camID[0],cd_p);

    tool.showPointCloud(cd_p);

    /**
     *  from XYZ to UV , show me the image
     *
     *  TODO
     *
     */


    /**
     *  fusing two image into a novel one.
     *
     *  show me the image.
     *
     *  TODO
     *
     * */


    return 0;
}


void test(Tool& tool)
{
    Matrix4d P;
    for(int i = 0;i <4;++i)
    {
        for(int j = 0 ; j < 4;++j)
        {
            P(i,j) = i*4+j;
        }
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cd( new pcl::PointCloud<pcl::PointXYZRGB>);
    cd->width = 200;
    cd->height = 100;
    cd->resize(cd->width * cd->height);

    for(int i = 0 ; i < cd->height; ++i)
    {
        for(int j = 0; j < cd->width; ++j)
        {
            cd->points[i*cd->width+j].x = 1024 * rand()/ (RAND_MAX+1.0f);
            cd->points[i*cd->width+j].y = 1024 * rand()/ (RAND_MAX+1.0f);
            cd->points[i*cd->width+j].z = 1024 * rand()/ (RAND_MAX+1.0f);
        }
    }

    Mat rgb;
    Mat depth;

    tool.projFromXYZToUV(cd,P,rgb,depth);

    imshow("rgb",rgb);
    imshow("depth",depth);

    waitKey(0);

}
