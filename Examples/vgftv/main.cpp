#include "tool.h"

using namespace std;
using namespace cv;
using namespace fvv_tool;
using namespace Eigen;


void test(Tool& );

int main(int argc, char ** argv)
{
    Tool tool;
    tool.loadImageParameter("./../../dataset/MSR3DVideo-Breakdancers/calibParams-breakdancers.txt");

//    tool.showParameter();

    tool.generateP(); // after loadImageParameter, generate each image's P according to reference camera 4

    // now each tool.cali[] has mP, cali[4]

    test(tool);

    /**
     * depth image operation, post-filtering
     *
     *  TODO
     *
     * **/

    /**
     *  project the nearest image to viewpoint
     *
     *  TODO
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
    pcl::PointCloud<pcl::PointXYZRGB> cd;
    cd.width = 200;
    cd.height = 100;
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

    Mat rgb;
    Mat depth;

    tool.projFromXYZToUV(cd,P,rgb,depth);

    imshow("rgb",rgb);
    imshow("depth",depth);

    waitKey(0);

}
