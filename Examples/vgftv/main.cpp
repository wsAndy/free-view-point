#include "tool.h"


using namespace std;
using namespace cv;
using namespace fvv_tool;
using namespace Eigen;


void test(Tool& );

// 在这个实验中，
//     object
//       ^
// 7-6-5-4-3-2-1-0
// right ---  left
int main(int argc, char ** argv)
{
    int camera_num = 8;
    Tool tool; // Tool tool("ballet",8);

    char* path_parameter = "/Users/sheng/Desktop/free-view-point/dataset/MSR3DVideo-Breakdancers/calibParams-breakdancers.txt";
    tool.loadImageParameter(path_parameter);
    tool.generateP(); // after loadImageParameter, generate each image's P according to reference camera 4

    //load camera's image into tool.cali
    string path = "/Users/sheng/Desktop/free-view-point/dataset/MSR3DVideo-Breakdancers/cam";

    std::vector<int> camID;

    camID.push_back(0);
    camID.push_back(1);
    camID.push_back(2);
    camID.push_back(5);
    camID.push_back(6);
    camID.push_back(7);

    tool.loadImage(path,camID,76,77);// image's startIndex = 0, endIndex = 1 defaultly . < 100

    // 这个是使用提供投影代码的测试
//    tool.forwardwarp(3, 4);
//    tool.forwardwarp(5, 4);

    tool.projUVtoXYZ(0,0,1);
    tool.projUVtoXYZ(1,0,1); // 0,1 in vector id
    tool.projUVtoXYZ(2,0,1);
    tool.projUVtoXYZ(5,0,1);
    tool.projUVtoXYZ(6,0,1);
    tool.projUVtoXYZ(7,0,1);

//    tool.writePLY("/Users/sheng/Desktop/free-view-point/pl3.ply",tool.cali[3].pl_vec[0]);
//    tool.writePLY("/Users/sheng/Desktop/free-view-point/pl5.ply",tool.cali[5].pl_vec[0]);

    // 设定目标位姿
    ImageFrame* cali = tool.getCamFrame();

    ImageFrame target_img;
    target_img.mP = cali[6].mP;
    target_img.RT = cali[6].RT;
    tool.projXYZtoUV(5,0,1,target_img);
    tool.projXYZtoUV(7,0,1,target_img);

    tool.smoothDepth(target_img,5); // 虽然在平滑之后的深度没有了部分突变，但是在原图投影过来的位置，依然是存在黑点，因此融合的结果中依然有黑点存在，因此需要使用反向warp

//    imwrite("/Users/sheng/Desktop/dep3.png",target_img.dep_vec[0]);
//    imwrite("/Users/sheng/Desktop/dep5.png",target_img.dep_vec[1]);

//    imwrite("/Users/sheng/Desktop/rgb3.jpg", target_img.rgb_vec[0]);
//    imwrite("/Users/sheng/Desktop/rgb5.jpg", target_img.rgb_vec[1]);
    tool.rendering(target_img);

    imwrite("/Users/sheng/Desktop/result.jpg",target_img.vir_img[0]);



    // 下面这堆是为了修正那个边缘，但是显然效果不好。
    Mat vir_rgb = target_img.vir_img[0];
    double left_u = 0.0, left_v = 0.0, right_u = 0.0, right_v = 0.0 ;

    Mat left_edge, right_edge;
    Mat left_dep_g,right_dep_g;
    Mat left_dep, right_dep;
    Mat left_rgb, right_rgb;

    left_dep = target_img.dep_vec[0];
    right_dep = target_img.dep_vec[1];
    left_rgb = cali[ target_img.proj_src_id[0] ].rgb_vec[0];
    right_rgb = cali[ target_img.proj_src_id[1] ].rgb_vec[0];

    Matrix4d target_mp = target_img.mP;
    Matrix4d left_mp = cali[ target_img.proj_src_id[0] ].mP;
    Matrix4d right_mp = cali[ target_img.proj_src_id[1] ].mP;

    cvtColor(left_dep, left_dep_g, CV_BGR2GRAY);
    cvtColor(right_dep, right_dep_g, CV_BGR2GRAY);
    Canny(left_dep_g,left_edge,10,255);
    Canny(right_dep_g,right_edge,10,255);

    imwrite("/Users/sheng/Desktop/left_edge.png",left_edge);

    for(int i = 0; i < left_edge.rows; ++i)
    {
        for(int j = 0; j < left_edge.cols; ++j)
        {
            if( left_edge.at<uchar>(i,j) > 0)
            {
                int sta_ind = 0;
                if( left_rgb.at<Vec3b>(i,j+1)[0] == 0 &&  left_rgb.at<Vec3b>(i,j+1)[1] == 0 && left_rgb.at<Vec3b>(i,j+1)[2] == 0 )
                {
                    sta_ind = -2;
                }else{
                    sta_ind = 0;
                }

                for(int ll = sta_ind; ll < sta_ind + 3; ++ll)
                {
                    int jj = j;
//                   直接去右图中寻找
                    jj = jj + ll;
                    tool.warpuv(target_mp, right_mp, (double)jj, (double)i, &right_u, &right_v, right_dep);
                    if( round(right_u) < 0 || round(right_u) > right_dep.cols || round(right_v) < 0 || round(right_v) >= right_dep.rows )
                    {
                        continue;
                    }

                    for(int ch = 0; ch < left_rgb.channels(); ++ch)
                    {
                        vir_rgb.at<Vec3b>(i,jj)[ch] = right_rgb.at<Vec3b>(round(right_v), round(right_u) )[ch];
                    }

                }
            }
        }
    }

    for(int i = 0; i < right_edge.rows; ++i)
    {
        for(int j = 0; j < right_edge.cols; ++j)
        {
            if( right_edge.at<uchar>(i,j) > 0 )
            {

                int sta_ind = 0;
                if( right_rgb.at<Vec3b>(i,j+1)[0] == 0 &&  right_rgb.at<Vec3b>(i,j+1)[1] == 0 && right_rgb.at<Vec3b>(i,j+1)[2] == 0 )
                {
                    sta_ind = -2;
                }else{
                    sta_ind = 0;
                }
                // 在水平的3个像素非0，投影到目标位置，记录，再投影到right位置，得到对应的颜色
                for(int ll = sta_ind; ll < sta_ind+ 3; ++ll)
                {
                    int jj = j;
//                   直接去右图中寻找
                    jj = jj + ll;
                    tool.warpuv(target_mp, left_mp, (double)jj, (double)i, &left_u, &left_v, left_dep);
                    if( round(left_u) < 0 || round(left_u) > left_dep.cols || round(left_v) < 0 || round(left_v) >= left_dep.rows )
                    {
                        continue;
                    }

                    for(int ch = 0; ch < left_rgb.channels(); ++ch)
                    {
                        vir_rgb.at<Vec3b>(i,jj)[ch] = left_rgb.at<Vec3b>(round(left_v), round(left_u) )[ch];
                    }

                }
            }
        }
    }

    imwrite("/Users/sheng/Desktop/result_2.jpg",vir_rgb);

    return 0;
}
