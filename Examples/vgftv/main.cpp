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

    std::vector<int> camID;

    camID.push_back(0);
    camID.push_back(1);

    tool.loadImage(path,camID);// image's startIndex = 0, endIndex = 1 defaultly

    /**
     *  from UV to XYZ , show me pcl_viewer
     *
     *  TODO
     *
     * **/

//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cd_p( new pcl::PointCloud<pcl::PointXYZRGB>);

//    tool.projFromUVToXYZ(tool.cali[camID[0]].rgb,tool.cali[camID[0]].dep,camID[0],cd_p);

//    tool.showPointCloud(cd_p);

    /**
     *  from XYZ to UV , show me the image
     *
     *  TODO
     *
     */
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cd_p( new pcl::PointCloud<pcl::PointXYZ>);

//    tool.projFromUVToXYZ(tool.cali[camID[0]].dep,camID[0],cd_p);

//    Mat target_dep;
//    std::vector<cv::Point> vir_link_ori;
//    tool.projFromXYZToUV(cd_p,tool.cali[camID[1]].mP,target_dep,vir_link_ori);

////    imshow("dep",target_dep);

//    Mat ori_dep_img = tool.cali[camID[1]].dep;
//    cvtColor(ori_dep_img,ori_dep_img,CV_BGR2GRAY);

//    resize(target_dep,target_dep,Size(ori_dep_img.cols,ori_dep_img.rows));

//    vector<Mat> output;
//    output.push_back(ori_dep_img); // B
//    output.push_back(Mat(Size(ori_dep_img.cols,ori_dep_img.rows),CV_8UC1));
//    output.push_back(target_dep); // R

//    Mat output_img;
//    merge(output,output_img);

//    imshow("fusing",output_img);

//    waitKey(0);


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

    std::vector<cv::Point> tmp_vv;
    tool.projFromXYZToUV(cd,P,rgb,depth,tmp_vv);

    imshow("rgb",rgb);
    imshow("depth",depth);

    waitKey(0);

}
