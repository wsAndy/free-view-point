#include "tool.h"


using namespace std;
using namespace cv;
using namespace fvv_tool;
using namespace Eigen;

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

    tool.loadImage(path,camID,18,19);// image's startIndex = 0, endIndex = 1 defaultly . < 100

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

    double mean_y = 0.0;
    for(int y_ind = 0; y_ind < 8; ++y_ind)
    {
        mean_y += cali[y_ind].pos(1);
    }
    mean_y = mean_y / 8.0;

//    Matrix3d K_mean;
//    K_mean = cali[0].K;
//    for(int k_ind = 1; k_ind < 8; ++k_ind)
//    {
//        K_mean = K_mean + cali[k_ind].K;
//    }
//    K_mean = K_mean / 8;

    // 设定目标位姿
    vector< vector<Vector3d>> pos_save;
    vector< vector<Vector3d>> om_save;

    int list[8] = {0,1,2,3,4,5,6,7};//,6,7};
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

//        cout << "left_cam_id = " << endl << cali[left_cam_id].pos << endl;
//        cout << "right_cam_id = " << endl << cali[right_cam_id].pos << endl;


        for(int ind = 1; ind < 10; ++ind)
        {
            Matrix3d now_R, now_R2;
            cv::Matx33d now_R_mat, now_R_mat2;
            Vector3d pos = left_pos+( right_pos - left_pos )*ind/(10), pos2 = left_pos+( right_pos - left_pos )*(ind+1)/(10);
            pos(1) = mean_y; pos2(1) = mean_y;
            Vector3d om_in = om*ind/10.0, om_in2 = om*(ind+1)/10.0;
            cv::Vec3d om_in_mat, om_in_mat2;


            om_in_mat(0) = om_in(0); om_in_mat(1) = om_in(1); om_in_mat(2) = om_in(2);
            Rodrigues(om_in_mat, now_R_mat);


            now_R(0,0) = now_R_mat(0,0);now_R(0,1) = now_R_mat(0,1);now_R(0,2) = now_R_mat(0,2);
            now_R(1,0) = now_R_mat(1,0);now_R(1,1) = now_R_mat(1,1);now_R(1,2) = now_R_mat(1,2);
            now_R(2,0) = now_R_mat(2,0);now_R(2,1) = now_R_mat(2,1);now_R(2,2) = now_R_mat(2,2);

            now_R = now_R * tmp_left_r;


            // ---
            om_in_mat2(0) = om_in2(0); om_in_mat2(1) = om_in2(1); om_in_mat2(2) = om_in2(2);
            Rodrigues(om_in_mat2, now_R_mat2);


            now_R2(0,0) = now_R_mat2(0,0);now_R2(0,1) = now_R_mat2(0,1);now_R2(0,2) = now_R_mat2(0,2);
            now_R2(1,0) = now_R_mat2(1,0);now_R2(1,1) = now_R_mat2(1,1);now_R2(1,2) = now_R_mat2(1,2);
            now_R2(2,0) = now_R_mat2(2,0);now_R2(2,1) = now_R_mat2(2,1);now_R2(2,2) = now_R_mat2(2,2);

            now_R2 = now_R2 * tmp_left_r;

            cv::Vec3d tmp_om_r1r2;
            cv::Matx33d r1r2;
            Matrix3d rr = now_R*now_R2.inverse();
            r1r2(0,0) = rr(0,0); r1r2(0,1) = rr(0,1); r1r2(0,2) = rr(0,2);
            r1r2(1,0) = rr(1,0); r1r2(1,1) = rr(1,1); r1r2(1,2) = rr(1,2);
            r1r2(2,0) = rr(2,0); r1r2(2,1) = rr(2,1); r1r2(2,2) = rr(2,2);

            Rodrigues( r1r2, tmp_om_r1r2);

            cout << "==========" <<list_ind << "=======" << endl;
            cout << "["<< ind << "] -> ["<<ind+1 << "]" << endl;
            cout << tmp_om_r1r2 << endl;




            Matrix3d K_mean = MatrixXd::Zero(3,3);


            double d1 =  tool.distance(cali[left_cam_id].pos, pos);
            double d2 =  tool.distance(cali[right_cam_id].pos, pos);

            K_mean = (d2/(d1+d2))*cali[left_cam_id].K + (d1/(d1+d2))*cali[right_cam_id].K;


        }


    }


    // 下面分析pos_save

    return 0;
}
