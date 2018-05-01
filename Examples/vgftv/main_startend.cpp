#include "tool.h"
#include "fstream"

using namespace std;
using namespace cv;
using namespace fvv_tool;
using namespace Eigen;

// 这个代码是直接拉0-7实现效果不好。
void test(Tool& );

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

//    double mean_y = 0.0;
//    for(int y_ind = 0; y_ind < 8; ++y_ind)
//    {
//        mean_y += cali[y_ind].pos(1);
//    }
//    mean_y = mean_y / 8.0;

//    Matrix3d K_mean;
//    K_mean = cali[0].K;
//    for(int k_ind = 1; k_ind < 8; ++k_ind)
//    {
//        K_mean = K_mean + cali[k_ind].K;
//    }
//    K_mean = K_mean / 8;


    // 设定目标位姿
    int list[8] = {0,1,2,3,4,5,6,7};
    for(int list_ind = 0; list_ind < 1; ++list_ind)
    {

        int left_cam_id = list_ind;
        int right_cam_id = list_ind+7;

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

        int ll_size = 80;
        for(int ind = 1; ind <= ll_size; ++ind)
        {

            Matrix3d now_R;
            Vector3d now_T;
            cv::Matx33d now_R_mat;
            Vector3d pos = left_pos+( right_pos - left_pos )*ind/(ll_size+1);

            Vector3d om_in = om*ind/(ll_size+1.0);
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


            // 下面这段是左右各一个
            int tmp_d1_ind = 0, tmp_d2_ind = 0;
            vector<double> d_positive, d_negative;
            vector<int> d_positive_ind, d_negative_ind;
            for(int kk = 0; kk < 8; ++kk)
            {
                double d = tool.distance(cali[kk].pos, pos);
                // 注意方向，但从[0]位置判别不出
                Vector4d zz1,zz2;
                zz1(0) = 0; zz2(0) = 0;
                zz1(1) = 0; zz2(1) = 0;
                zz1(2) = 0; zz2(2) = 10;
                zz1(3) = 1; zz2(3) = 1;

                Vector4d cali_pos_zz1 = tool.changeRT(cali[kk].RT) * zz1;
                Vector4d cali_pos_zz2 = tool.changeRT(cali[kk].RT) * zz2;
                Vector4d pos_zz1 = tool.changeRT(rt) * zz1;

                double x0 = cali_pos_zz1(0);
                double y0 = cali_pos_zz1(2);
                double x1 = cali_pos_zz2(0);
                double y1 = cali_pos_zz2(2);
                double px = pos_zz1(0);
                double py = pos_zz1(2);

                double tmp = ( (x1 - x0)*(py - y0) - (px - x0)*(y1 - y0) );
                if( tmp > 0 )
                {
                    d_positive.push_back(d);
                    d_positive_ind.push_back(kk);
                }else{
                    d_negative.push_back(d);
                    d_negative_ind.push_back(kk);
                }
            }

            vector<double> d_positive_tmp, d_negative_tmp;
            d_positive_tmp = d_positive;
            d_negative_tmp = d_negative;

            sort(d_positive_tmp.begin(), d_positive_tmp.end());
            sort(d_negative_tmp.begin(), d_negative_tmp.end());

            tmp_d1_ind = d_positive_ind[find(d_positive.begin(), d_positive.end(), d_positive_tmp[0] ) - d_positive.begin()];
            tmp_d2_ind = d_negative_ind[find(d_negative.begin(), d_negative.end(), d_negative_tmp[0] ) - d_negative.begin()];

            if( d_positive_tmp[0] > d_negative_tmp[0] )
            {
                tmp_d2_ind = tmp_d2_ind + tmp_d1_ind;
                tmp_d1_ind = tmp_d2_ind - tmp_d1_ind;
                tmp_d2_ind = tmp_d2_ind - tmp_d1_ind;
            }

            // tmp_d1_ind是最小的 tmp_d2_ind是第二小的编号

            double d1 =  tool.distance(cali[tmp_d1_ind].pos, pos);// d1一定最小
            double d2 =  tool.distance(cali[tmp_d2_ind].pos, pos);// d2一定第二小

            K_mean = (d2/(d1+d2))*cali[tmp_d1_ind].K + (d1/(d1+d2))*cali[tmp_d2_ind].K;

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
            tool.projXYZtoUV(tmp_d1_ind,0,1,target_img);
            tool.projXYZtoUV(tmp_d2_ind,0,1,target_img);

//            if(ind == 5)
//            {
//                imwrite("/Users/sheng/Desktop/img/5_0.jpg",target_img.rgb_vec[0]);
//                imwrite("/Users/sheng/Desktop/img/5_1.jpg",target_img.rgb_vec[1]);
//            }

//            if(ind == 6)
//            {
//                imwrite("/Users/sheng/Desktop/img/6_1.jpg",target_img.rgb_vec[0]);
//                imwrite("/Users/sheng/Desktop/img/6_0.jpg",target_img.rgb_vec[1]);
//            }

//            tool.smoothDepth(target_img,3); // 虽然在平滑之后的深度没有了部分突变，但是在原图投影过来的位置，依然是存在黑点，因此融合的结果中依然有黑点存在，因此需要使用反向warp

            tool.renderingTest(target_img,d1,d2);

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

            ss_str.clear();
            ss.clear();
            tool.releaseImageFrame(target_img);
        }
    }

    ou.close();

    return 0;
}
