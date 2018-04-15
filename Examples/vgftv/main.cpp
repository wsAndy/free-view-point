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
    tool.generateP(); // after loadImageParameter, generate each image's P according to reference camera 4

    //load camera's image into tool.cali
    string path = "./../../dataset/MSR3DVideo-Breakdancers/cam";

    std::vector<int> camID;

    camID.push_back(0);
    camID.push_back(3);
    camID.push_back(5);
    camID.push_back(6);

    tool.loadImage(path,camID,75,76);// image's startIndex = 0, endIndex = 1 defaultly . < 100

    /**
     *  from UV to XYZ , show me pcl_viewer
     *
     *  TODO
     *
     * **/

    // 对0号相机的存储图像的[0,1)位置序列投影到XYZ中
    tool.projUVtoXYZ(0,0,1);
    tool.projUVtoXYZ(3,0,1); // 0,1 in vector id
    tool.projUVtoXYZ(5,0,1);
    tool.projUVtoXYZ(6,0,1);
    tool.writePLY("/Users/sheng/Desktop/free-view-point/pl3.ply",tool.cali[3].pl_vec[0]);
    tool.writePLY("/Users/sheng/Desktop/free-view-point/pl5.ply",tool.cali[5].pl_vec[0]);


    /**
     *  from XYZ to UV , show me the image
     *
     *  TODO
     *
     */
    // 设定目标位姿
    ImageFrame* cali = tool.getCamFrame();

    ImageFrame target_img;
    target_img.mP = cali[4].mP;
    target_img.RT = cali[4].RT;
    tool.projXYZtoUV(3,0,1,target_img);

//    target_img.mP = cali[4].mP;
//    target_img.RT = cali[4].RT;
    tool.projXYZtoUV(5,0,1,target_img);

//    target_img.mP = cali[5].mP;
//    target_img.RT = cali[5].RT;
//    tool.projXYZtoUV(0,0,1,target_img);

//    imwrite("/Users/sheng/Desktop/dep3.png",target_img.dep_vec[0]);
    imwrite("/Users/sheng/Desktop/dep3.png",target_img.dep_vec[0]);
    imwrite("/Users/sheng/Desktop/dep5.png",target_img.dep_vec[1]);

//    imwrite("/Users/sheng/Desktop/rgb3.jpg", target_img.rgb_vec[0]);
    imwrite("/Users/sheng/Desktop/rgb3.jpg", target_img.rgb_vec[0]);
    imwrite("/Users/sheng/Desktop/rgb5.jpg", target_img.rgb_vec[1]);

//    imshow("ori", cali[4].rgb_vec[0]);
//    imshow("4",target_img.rgb_vec[0]);
//    waitKey(0);


    /**
     *  fusing two image into a novel one.
     *
     *  show me the image.
     *
     *  TODO
     *
     * */

    Matrix4d target_P = tool.cali[4].mP;

//    tool.rendering(camID,target_P);

    return 0;
}


//void test(Tool& tool)
//{
//    Matrix4d P;
//    for(int i = 0;i <4;++i)
//    {
//        for(int j = 0 ; j < 4;++j)
//        {
//            P(i,j) = i*4+j;
//        }
//    }
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cd( new pcl::PointCloud<pcl::PointXYZRGB>);
//    cd->width = 200;
//    cd->height = 100;
//    cd->resize(cd->width * cd->height);

//    for(int i = 0 ; i < cd->height; ++i)
//    {
//        for(int j = 0; j < cd->width; ++j)
//        {
//            cd->points[i*cd->width+j].x = 1024 * rand()/ (RAND_MAX+1.0f);
//            cd->points[i*cd->width+j].y = 1024 * rand()/ (RAND_MAX+1.0f);
//            cd->points[i*cd->width+j].z = 1024 * rand()/ (RAND_MAX+1.0f);
//        }
//    }

//    Mat rgb;
//    Mat depth;

//    std::vector<cv::Point> tmp_vv;
//    tool.projFromXYZToUV(cd,P,rgb,depth,tmp_vv);

//    imshow("rgb",rgb);
//    imshow("depth",depth);

//    waitKey(0);

//}
