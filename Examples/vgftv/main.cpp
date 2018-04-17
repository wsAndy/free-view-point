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
    camID.push_back(3);
    camID.push_back(4);
    camID.push_back(5);
    camID.push_back(6);
    camID.push_back(7);

    tool.loadImage(path,camID,0,1);// image's startIndex = 0, endIndex = 1 defaultly . < 100

    // 这个是使用提供投影代码的测试
//    tool.forwardwarp(3, 4);
//    tool.forwardwarp(5, 4);

    tool.projUVtoXYZ(0,0,1);
    tool.projUVtoXYZ(1,0,1); // 0,1 in vector id
    tool.projUVtoXYZ(2,0,1);
    tool.projUVtoXYZ(3,0,1);
    tool.projUVtoXYZ(4,0,1);
    tool.projUVtoXYZ(5,0,1);
    tool.projUVtoXYZ(6,0,1);
    tool.projUVtoXYZ(7,0,1);

    tool.getFrontBackGround(0,0,1);
    tool.getFrontBackGround(1,0,1);
    tool.getFrontBackGround(2,0,1);
    tool.getFrontBackGround(3,0,1);
    tool.getFrontBackGround(4,0,1);
    tool.getFrontBackGround(5,0,1);
    tool.getFrontBackGround(6,0,1);
    tool.getFrontBackGround(7,0,1);

//    tool.writePLY("/Users/sheng/Desktop/free-view-point/pl3.ply",tool.cali[3].pl_vec[0]);
//    tool.writePLY("/Users/sheng/Desktop/free-view-point/pl5.ply",tool.cali[5].pl_vec[0]);

    ImageFrame* cali = tool.getCamFrame();

    // 设定目标位姿
    int list[8] = {0,1,2,3,4,5,6,7};
    for(int list_ind = 0; list_ind < 7; ++list_ind)
    {
        int left_cam_id = list[list_ind];
        int right_cam_id = list[list_ind+1];

        Matrix4d tmp_left_rt, tmp_right_rt;
        Matrix3d tmp_left_r, tmp_right_r;
        Vector3d tmp_left_t,tmp_right_t;

        tmp_left_rt = cali[left_cam_id].RT;
        tmp_right_rt = cali[right_cam_id].RT;
        tmp_left_r = tmp_left_rt.block<3,3>(0,0);
        tmp_left_t = tmp_left_rt.block<3,1>(0,3);

        tmp_right_r = tmp_right_rt.block<3,3>(0,0);
        tmp_right_t = tmp_right_rt.block<3,1>(0,3);

        Vector3d left_pos = -1* tmp_left_r.inverse() * tmp_left_t;
        Vector3d right_pos = -1* tmp_right_r.inverse() * tmp_right_t;

        Matrix3d tmp_r1r2;
        tmp_r1r2 = tmp_right_r * tmp_left_r.inverse();


        cv::Matx33d tmp_R;
        cv::Vec3d tmp_om;
        tmp_R(0,0) = tmp_r1r2(0,0);tmp_R(0,1) = tmp_r1r2(0,1);tmp_R(0,2) = tmp_r1r2(0,2);
        tmp_R(1,0) = tmp_r1r2(1,0);tmp_R(1,1) = tmp_r1r2(1,1);tmp_R(1,2) = tmp_r1r2(1,2);
        tmp_R(2,0) = tmp_r1r2(2,0);tmp_R(2,1) = tmp_r1r2(2,1);tmp_R(2,2) = tmp_r1r2(2,2);


        Rodrigues(tmp_R, tmp_om);
        Vector3d om;
        om(0) = tmp_om(0);
        om(1) = tmp_om(1);
        om(2) = tmp_om(2);

//        cout << "om = " << om << endl;

        for(int ind = 1; ind < 10; ++ind)
        {
            Matrix3d now_R;
            Vector3d now_T;
            cv::Matx33d now_R_mat;
            Vector3d pos = left_pos+( right_pos - left_pos )*ind/(10);
            Vector3d om_in = om*ind/10.0;
            cv::Vec3d om_in_mat;

            om_in_mat(0) = om_in(0); om_in_mat(1) = om_in(1); om_in_mat(2) = om_in(2);
            Rodrigues(om_in_mat, now_R_mat);

//            cout << now_R_mat << endl;

            now_R(0,0) = now_R_mat(0,0);now_R(0,1) = now_R_mat(0,1);now_R(0,2) = now_R_mat(0,2);
            now_R(1,0) = now_R_mat(1,0);now_R(1,1) = now_R_mat(1,1);now_R(1,2) = now_R_mat(1,2);
            now_R(2,0) = now_R_mat(2,0);now_R(2,1) = now_R_mat(2,1);now_R(2,2) = now_R_mat(2,2);

            now_R = now_R * tmp_left_r;

            now_T = -1*now_R*pos;

            Matrix4d rt;
            rt.block<3,3>(0,0) = now_R;
            rt.block<3,1>(0,3) = now_T;
            rt(3,0) = 0;
            rt(3,1) = 0;
            rt(3,2) = 0;
            rt(3,3) = 1;

            Matrix4d mp;
            mp.block<3,3>(0,0) = cali[0].K * now_R;
            mp.block<3,1>(0,3) = cali[0].K * now_T;
            mp(3,0) = 0;
            mp(3,1) = 0;
            mp(3,2) = 0;
            mp(3,3) = 1;

//            cout << "----"<< ind <<"-----" << endl;
//            cout << "left_rt" << endl;
//            cout << cali[0].RT << endl;
//            cout << "pos_r" << endl;
//            cout << now_R << endl;
//            cout << "pos_t" << endl;
//            cout << now_T << endl;
//            cout << "========" << endl;


            ImageFrame target_img;
            target_img.mP = mp;
            target_img.RT = rt;
            tool.projXYZtoUV(left_cam_id,0,1,target_img);
            tool.projXYZtoUV(right_cam_id,0,1,target_img);

            tool.smoothDepth(target_img,3); // 虽然在平滑之后的深度没有了部分突变，但是在原图投影过来的位置，依然是存在黑点，因此融合的结果中依然有黑点存在，因此需要使用反向warp

            tool.rendering(target_img);

            stringstream ss;
            ss << "/Users/sheng/Desktop/img/";
            ss << left_cam_id;
            ss << "_";
            ss << right_cam_id;
            ss << "_";
            ss << ind;
            ss << ".jpg";
            string ss_str;
            ss >> ss_str;
            imwrite(ss_str, target_img.vir_img[0]);

            ss_str.clear();
            ss.clear();
            tool.releaseImageFrame(target_img);
        }
    }

    return 0;
}
