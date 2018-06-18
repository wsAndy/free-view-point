#include "tool.h"
#include "fstream"

using namespace std;
using namespace cv;
using namespace fvv_tool;
using namespace Eigen;

// 本文件实验 View Synthesis for advanced 3D Video Systems中，同样也是上交之前文章的分层投影方法。
// 从投影的结果上看，先删除主投影的背景边缘（深度突变），再用次投影的响应位置的补上，效果能改善。

// 在这个实验中，
//     object
//       ^
// 7-6-5-4-3-2-1-0
// right ---  left
int main(int argc, char ** argv)
{

    ofstream ou;
    ou.open("/Users/sheng/Desktop/pos/pos.txt");



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

    tool.getFrontBackGround(0,0,1);
    tool.getFrontBackGround(1,0,1);
    tool.getFrontBackGround(2,0,1);
    tool.getFrontBackGround(3,0,1);
    tool.getFrontBackGround(4,0,1);
    tool.getFrontBackGround(5,0,1);
    tool.getFrontBackGround(6,0,1);
    tool.getFrontBackGround(7,0,1);

    tool.projUVtoXYZ(0,0,1);
    tool.projUVtoXYZ(1,0,1); // 0,1 in vector id
    tool.projUVtoXYZ(2,0,1);
    tool.projUVtoXYZ(3,0,1);
    tool.projUVtoXYZ(4,0,1);
    tool.projUVtoXYZ(5,0,1);
    tool.projUVtoXYZ(6,0,1);
    tool.projUVtoXYZ(7,0,1);


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

        for(int ind = 0; ind <= 10; ++ind)
        {

            Matrix3d now_R;
            Vector3d now_T;
            cv::Matx33d now_R_mat;
            Vector3d pos = left_pos+( right_pos - left_pos )*ind/(10);

            Vector3d om_in = om*ind/10.0;
            cv::Vec3d om_in_mat;

            om_in_mat(0) = om_in(0); om_in_mat(1) = om_in(1); om_in_mat(2) = om_in(2);
            Rodrigues(om_in_mat, now_R_mat);

            now_R(0,0) = now_R_mat(0,0);now_R(0,1) = now_R_mat(0,1);now_R(0,2) = now_R_mat(0,2);
            now_R(1,0) = now_R_mat(1,0);now_R(1,1) = now_R_mat(1,1);now_R(1,2) = now_R_mat(1,2);
            now_R(2,0) = now_R_mat(2,0);now_R(2,1) = now_R_mat(2,1);now_R(2,2) = now_R_mat(2,2);

            now_R = now_R * tmp_left_r;

            now_T = -1*now_R*pos;


//            cout << "now_T = " << now_T << endl; // 感觉上面个增稳是错误的，因为基本无效，而且上面只是在平均，这边才是实际增加

//            now_T(0) = now_T(0) + 4; // 镜头右移-》偏向left右侧！
//            now_T(1) = now_T(1) + 4; // 镜头下移
//            now_T(2) = now_T(2) + 10; // 镜头后移
// 可以判断是左手坐标系
//            now_T(2) = now_T(2) - 10;

//            now_T(1) = 0; //========

            Matrix4d rt;
            rt.block<3,3>(0,0) = now_R;
            rt.block<3,1>(0,3) = now_T;
            rt(3,0) = 0;
            rt(3,1) = 0;
            rt(3,2) = 0;
            rt(3,3) = 1;

            ou << now_R(0,0) << " " <<  now_R(0,1) << " "  << now_R(0,2) << endl;
            ou << now_R(1,0) << " " <<  now_R(1,1) << " "  << now_R(1,2) << endl;
            ou << now_R(2,0) << " " <<  now_R(2,1) << " "  << now_R(2,2) << endl;
            ou << now_T(0) << " " <<  now_T(1) << " "  << now_T(2) << endl;


            Matrix3d K_mean = MatrixXd::Zero(3,3);


            double d1 =  tool.distance(cali[left_cam_id].pos, pos);
            double d2 =  tool.distance(cali[right_cam_id].pos, pos);

            K_mean = (d2/(d1+d2))*cali[left_cam_id].K + (d1/(d1+d2))*cali[right_cam_id].K;

            Matrix4d mp;
            mp.block<3,3>(0,0) = K_mean * now_R;
            mp.block<3,1>(0,3) = K_mean * now_T;
            mp(3,0) = 0;
            mp(3,1) = 0;
            mp(3,2) = 0;
            mp(3,3) = 1;

            ImageFrame target_img;
            target_img.mP = mp;
            target_img.RT = rt;
            if( d1 < d2 )
            {
                tool.projXYZtoUV(left_cam_id,0,1,target_img,true);
                tool.projXYZtoUV(right_cam_id,0,1,target_img,false);
            }else{
                tool.projXYZtoUV(left_cam_id,0,1,target_img, false);
                tool.projXYZtoUV(right_cam_id,0,1,target_img, true);
            }

            // 这个里面只对left做了处理
            tool.getProjBackground(target_img,d1,d2);

// 这个函数不靠谱，实验结果发现产生了更多的洞！
//            tool.smoothDepth(target_img,3);
//            tool.rendering_backward(target_img,d1,d2);
            tool.rendering_forward(target_img,d1,d2);
            // 融合之后的结果中，由于在一开始getProjBackground中删除部分点会导致一些凸出的尖峰状干扰，建议在周围做局部的均值操作（在最外围做区域的平均）
            // 但是getProjBackground里面那个back通过target_img拿出来总是会出错，这个是一个bug

            // 对rendering之后的结果，进行黑洞的补全
            tool.repair(target_img);


            stringstream ss;
            ss << "/Users/sheng/Desktop/img/";
            ss << left_cam_id;
            ss << "_";
            ss << right_cam_id;
            ss << "_";
            ss << ind;
            ss << "test.jpg";
            string ss_str;
            ss >> ss_str;
            imwrite(ss_str, target_img.vir_img[0]);

            ss.clear();
            ss_str.clear();
            ss << "/Users/sheng/Desktop/left_pro/";
            ss << left_cam_id;
            ss << "_";
            ss << right_cam_id;
            ss << "_";
            ss << ind;
            ss << "_left.jpg";
            ss >> ss_str;
            imwrite(ss_str, target_img.rgb_vec[0]);

            ss.clear();
            ss_str.clear();
            ss << "/Users/sheng/Desktop/right_pro/";
            ss << left_cam_id;
            ss << "_";
            ss << right_cam_id;
            ss << "_";
            ss << ind;
            ss << "_right.jpg";
            ss >> ss_str;
            imwrite(ss_str, target_img.rgb_vec[1]);

            ss_str.clear();
            ss.clear();
            tool.releaseImageFrame(target_img);
        }
    }

    ou.close();

    return 0;
}
