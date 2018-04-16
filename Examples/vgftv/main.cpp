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

    // 这个是使用提供投影代码的测试
    tool.forwardwarp(3, 4);
//    tool.forwardwarp(5, 4);

    tool.projUVtoXYZ(0,0,1);
    tool.projUVtoXYZ(3,0,1); // 0,1 in vector id
    tool.projUVtoXYZ(5,0,1);
    tool.projUVtoXYZ(6,0,1);
    tool.writePLY("/Users/sheng/Desktop/free-view-point/pl3.ply",tool.cali[3].pl_vec[0]);
    tool.writePLY("/Users/sheng/Desktop/free-view-point/pl5.ply",tool.cali[5].pl_vec[0]);

    // 设定目标位姿
    ImageFrame* cali = tool.getCamFrame();

    ImageFrame target_img;
    target_img.mP = cali[4].mP;
    target_img.RT = cali[4].RT;
    tool.projXYZtoUV(3,0,1,target_img);
    tool.projXYZtoUV(5,0,1,target_img);

    tool.smoothDepth(target_img,3); // 虽然在平滑之后的深度没有了部分突变，但是在原图投影过来的位置，依然是存在黑点，因此融合的结果中依然有黑点存在，因此需要使用反向warp

//    imwrite("/Users/sheng/Desktop/dep3.png",target_img.dep_vec[0]);
//    imwrite("/Users/sheng/Desktop/dep5.png",target_img.dep_vec[1]);

//    imwrite("/Users/sheng/Desktop/rgb3.jpg", target_img.rgb_vec[0]);
//    imwrite("/Users/sheng/Desktop/rgb5.jpg", target_img.rgb_vec[1]);
    tool.rendering(target_img);

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
