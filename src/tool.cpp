#include "tool.h"

using namespace std;
using namespace fvv_tool;
using namespace cv;

Tool::Tool()
{
    cout << "Tool init." <<endl;
    cali = new ImageFrame[camera_num];
}

Tool::Tool(string dataset_name)
{
    if( dataset_name.compare("ballet") == 0)
    {
        MaxZ = 130;
        MinZ = 42;
    }else{
        MaxZ = 120;
        MinZ = 44;
    }

    cali = new ImageFrame[camera_num];
}

Tool::~Tool()
{
    cout << "Tool clean" <<endl;
}



void Tool::showParameter()
{
    for(int k = 0; k < camera_num; ++k)
    {
        cout << "-----------"<<endl;
        cout << "camera " << k <<endl;
        cout << "camera intrinsic = " << endl;
        for(int i = 0; i < 3; ++i)
        {
            for(int j = 0; j < 3; ++j)
            {
                cout << cali->mK[i][j] << "  ";
            }
            cout << endl;
        }
        cout << "camera Extrinsic = " << endl;
        for(int i = 0; i< 3; ++i)
        {
            for(int j = 0; j < 3; ++j)
            {
                cout << cali->mR[i][j] << "  ";
            }

            cout << cali->mT[i] <<endl;
        }

    }

}


/*
*   load camera calibration parameter, the parameter in the following format:
*
*  CameraNumber
*  Intrinsic matrix
*  barrel distortion (can be ignored, since its always 0, 0)
*  Extrinsic Matrix
*
*
*/
void Tool::loadImageParameter(char* file_name)
{
//    fstream f_read;
//    f_read.open(file_name, ios::in);
    FILE *f_read;
    f_read = fopen(file_name,"r");

    if(!f_read)
    {
        cerr << "ERROR! [" << file_name <<"] not exist."<<endl;
    }else{
        cerr << "load calibration parameter" <<endl;

        int camIdx;
        double tmp; // dummy
        for(int k = 0; k < camera_num; ++k)
        {
            fscanf(f_read,"%d",&camIdx);

            // camera intrinsics
            for (int i = 0; i < 3; ++i)
            {
                fscanf(f_read,"%lf\t%lf\t%lf",
                       &(cali[camIdx].mK[i][0]),
                       &(cali[camIdx].mK[i][1]),
                       &(cali[camIdx].mK[i][2]));
            }

            fscanf(f_read,"%lf",&tmp);
            fscanf(f_read,"%lf",&tmp);

            // camera rotation and transformation
            for (int i = 0; i < 3; ++i)
            {

                for(int j = 0; j < 3; ++j)
                {
                    fscanf(f_read,"%lf",
                           &(cali[camIdx].mR[i][j]));
                }

                fscanf(f_read,"%lf",
                       &(cali[camIdx].mT[i]));

            }
        }

        fclose(f_read);

    }

}


/*
*   use Eigen to calculate P matrix
*
**/
void Tool::generateP()
{
    for(int k = 0; k < camera_num; ++k)
    {

        Matrix3d eg_mk, eg_mr;
        Vector3d eg_mt;
        for(int i =0; i<3; ++i)
        {
            for(int j = 0; j < 3;++j)
            {
                eg_mk(i,j) = cali[k].mK[i][j];
                eg_mr(i,j) = cali[k].mR[i][j];
            }
            eg_mt(i) = cali[k].mT[i];
        }

        Matrix4d eg_P;
        eg_P.block<3,3>(0,0) = eg_mk*eg_mr;
        eg_P.block<3,1>(0,3) = eg_mk*eg_mt;
        eg_P(3,0) = 0.0;
        eg_P(3,1) = 0.0;
        eg_P(3,2) = 0.0;
        eg_P(3,3) = 1.0;

        cali[k].mP = eg_P;
    }

}


double Tool::getPixelActualDepth(unsigned char d)
{

/*
    Mat ac_dep(depth.cols,depth.rows,CV_8UC1); // in this dataset, depth image is saved in 8bits.

    vector<Mat> spl;
    split(depth,spl);

    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> eg_dep,eg_out;
    cv2eigen(spl[0],eg_dep);

    eg_dep = (eg_dep/255)*(1.0/MinZ - 1.0/MaxZ);
    // here you also need to operate each element with  +(1/MaxZ) , and get 1/Z

    eigen2cv(eg_out,ac_dep);
*/

    return 1.0/((d/255.0)*(1.0/MinZ - 1.0/MaxZ) + 1.0/MaxZ);
}

double Tool::getPixelDepth(double dw)
{
    return 255.0*(MaxZ*MinZ)*(1/dw-1/MaxZ)/(MaxZ-MinZ);
}

void Tool::rendering(vector<int> &img_id, Matrix4d& targetP)
{
    // use two image

    Mat vir_depth; // depth image in novel viewpoint
    Mat vir_rgb;   // rgb image in novel viewpoint

    Mat left_d = cali[img_id[0]].dep;
    Mat right_d = cali[img_id[1]].dep;

    Mat left_r = cali[img_id[0]].rgb;
    Mat right_r = cali[img_id[1]].rgb;

    Matrix4d left_P = cali[img_id[0]].mP;
    Matrix4d right_P = cali[img_id[1]].mP;

    pcl::PointCloud<pcl::PointXYZ> tmp_l_cd;
    pcl::PointCloud<pcl::PointXYZ> tmp_r_cd;


    /**
     * project left depth image to virtual image plane
     * project right depth image to virtual image plane
     * fuseing these two depth images and get one.
     *
     **/

    Mat left_vir_d; // left project to virtual depth image.

    projFromUVToXYZ(left_d,img_id[0],tmp_l_cd);
    projFromXYZToUV(tmp_l_cd,targetP,left_vir_d);

    Mat right_vir_d; // right project to virtual depth image

    projFromUVToXYZ(right_d,img_id[0],tmp_r_cd);
    projFromXYZToUV(tmp_r_cd,targetP,right_vir_d);

    fusingDepth(left_vir_d,right_vir_d,vir_depth);

    // vir_depth  is already.

    /**
     *  you can smooth the depth image here.
     *
     */

    // TODO

    /**
     *  reproject the pixel on virtual image plane to left image and right image
     *  get its rgb values
     *
     */

    // TODO


}

/**
 *  fusing two depth image
 *
 */

void Tool::fusingDepth(Mat &left, Mat &right, Mat &target)
{
    if(left.cols!=right.cols || left.rows!= right.rows)
    {
        cerr << "Error! in function [fusingDepth], input left image and right image not in the same size." <<endl;
    }else{

        target = cv::Mat::zeros(left.rows,left.cols,CV_8UC1);

        int count = 0;
        for(int i = 0; i < left.rows; ++i)
        {
            for(int j = 0; j < left.cols; ++j)
            {
                if (left.at<uchar>(i,j) < 1 && right.at<uchar>(i,j) < 1)
                { // both not have value
                    count = count + 1; // count those empty point
                  continue; // stay 0
                }else if ( left.at<uchar>(i,j) < 1 && right.at<uchar>(i,j) >=1 )
                {
                    target.at<uchar>(i,j) = right.at<uchar>(i,j);

                }else if (left.at<uchar>(i,j) >= 1 && right.at<uchar>(i,j) < 1)
                {
                    target.at<uchar>(i,j) = left.at<uchar>(i,j);

                }else{
                    // both has value

                    target.at<uchar>(i,j) = min(left.at<uchar>(i,j),right.at<uchar>(i,j));
                }
            }

        }

        cerr << "In fusing Depth image, " << (count)/(left.rows * left.cols)
             << "% points is empty in virtual image" << endl;
    }

}




/**
 *  here need to accelerate
 *
 *  project from UV (image coordinate ) to XYZ (3D space)
 *
 *
 *  Z_w * x = P * X
 *      UV       XYZ
 *
 *  input: rgb , dep
 *  output: cd_
 **/

void Tool::projFromUVToXYZ(Mat &rgb, Mat &dep, int index, pcl::PointCloud<pcl::PointXYZRGB> &cd_)
{
    cd_.width = rgb.cols;
    cd_.height = rgb.rows;
    cd_.points.resize(cd_.width * cd_.height);

    for(int i = 0; i < cd_.height; ++i)
    {
        for(int j = 0; j < cd_.width; ++j)
        {
            double zc = getPixelActualDepth(dep.at<cv::Vec3b>(i,j)[0]); // actual depth
            int u = j;
            int v = i;

            Matrix4d p_in = (cali[index].mP).inverse();
            Vector4d x_(zc*u,zc*v,zc,1);
            Vector4d X_;
            X_ = p_in*x_;

            cd_.points[i*cd_.width+j].x = X_(0);
            cd_.points[i*cd_.width+j].y = X_(1);
            cd_.points[i*cd_.width+j].z = X_(2);

            cd_.points[i*cd_.width+j].r = rgb.at<cv::Vec3b>(i,j)[2];
            cd_.points[i*cd_.width+j].g = rgb.at<cv::Vec3b>(i,j)[1];
            cd_.points[i*cd_.width+j].b = rgb.at<cv::Vec3b>(i,j)[0];

        }

    }

}


/**
 *  here need to accelerate
 *
 *  project from UV (image coordinate ) to XYZ (3D space)
 *
 *  only project depth
 *
 **/

void Tool::projFromUVToXYZ( Mat &dep, int index, pcl::PointCloud<pcl::PointXYZ> &cd_)
{
    cd_.width = dep.cols;
    cd_.height = dep.rows;
    cd_.points.resize(cd_.width * cd_.height);

    for(int i = 0; i < cd_.height; ++i)
    {
        for(int j = 0; j < cd_.width; ++j)
        {
            double zc = getPixelActualDepth(dep.at<cv::Vec3b>(i,j)[0]); // actual depth
            int u = j;
            int v = i;

            Matrix4d p_in = (cali[index].mP).inverse();
            Vector4d x_(zc*u,zc*v,zc,1);
            Vector4d X_;
            X_ = p_in*x_;

            cd_.points[i*cd_.width+j].x = X_(0);
            cd_.points[i*cd_.width+j].y = X_(1);
            cd_.points[i*cd_.width+j].z = X_(2);

        }

    }

}

/**
 *  project from XYZ (3D world coordinate) to UV (image coordinate)
 *
 *  input: cd_
 *  output: rgb, dep
 */

void Tool::projFromXYZToUV(pcl::PointCloud<pcl::PointXYZRGB> &cd_, Eigen::Matrix4d & targetP, Mat &rgb, Mat &dep)
{

    // here you need to initial rgb and depth first since not all the pixel in these two image will be fixed.

    if(targetP.cols()!=4 || targetP.rows()!=4)
    {
        cerr << " targetP is not a [4x4] matrix!" <<endl;
    }else{
        // initial rgb and depth
        // TODO

        rgb = Mat::zeros(cd_.height,cd_.width,CV_8UC3);
        dep = Mat::zeros(cd_.height,cd_.width,CV_8UC1);

        for(int i = 0; i < cd_.height; ++i)
        {
            for(int j = 0 ; j < cd_.width; ++j)
            {
                Vector4d X_;
                X_(0) = cd_.points[i*cd_.width+j].x;
                X_(1) = cd_.points[i*cd_.width+j].y;
                X_(2) = cd_.points[i*cd_.width+j].z; // actual depth
                X_(3) = 1.0;

                double zc = X_(2);
                Vector4d x_;
                x_ = targetP*X_;

                if(zc < 0.2) // important in test.
                {
                    continue;
                }

                rgb.at<cv::Vec3b>(int(x_(1)/zc),int(x_(0)/zc))[0] = cd_.points[i*cd_.width+j].b;
                rgb.at<cv::Vec3b>(int(x_(1)/zc),int(x_(0)/zc))[1] = cd_.points[i*cd_.width+j].g;
                rgb.at<cv::Vec3b>(int(x_(1)/zc),int(x_(0)/zc))[2] = cd_.points[i*cd_.width+j].r;

                dep.at<uchar>(int(x_(1)/zc),int(x_(0)/zc)) = getPixelDepth(zc);
            }
        }
    }

}



/**
 *  project from XYZ (3D world coordinate) to UV (image coordinate)
 *
 *  input: cd_
 *  output: dep
 *
 *   only depth image
 */

void Tool::projFromXYZToUV(pcl::PointCloud<pcl::PointXYZ> &cd_, Eigen::Matrix4d & targetP, Mat &dep)
{

    // here you need to initial rgb and depth first since not all the pixel in these two image will be fixed.

    if(targetP.cols()!=4 || targetP.rows()!=4)
    {
        cerr << " targetP is not a [4x4] matrix!" <<endl;
    }else{
        // initial depth
        // TODO

        dep = Mat::zeros(cd_.height,cd_.width,CV_8UC1);

        for(int i = 0; i < cd_.height; ++i)
        {
            for(int j = 0 ; j < cd_.width; ++j)
            {
                Vector4d X_;
                X_(0) = cd_.points[i*cd_.width+j].x;
                X_(1) = cd_.points[i*cd_.width+j].y;
                X_(2) = cd_.points[i*cd_.width+j].z; // actual depth
                X_(3) = 1.0;

                double zc = X_(2);
                Vector4d x_;
                x_ = targetP*X_;

                if(zc < 0.2) // important in test.
                {
                    continue;
                }

                dep.at<uchar>(int(x_(1)/zc),int(x_(0)/zc)) = getPixelDepth(zc);
            }
        }
    }

}
