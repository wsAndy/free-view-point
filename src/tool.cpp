#include "tool.h"

using namespace std;
using namespace fvv_tool;
using namespace cv;

Tool::Tool()
{
    cout << "Tool init." <<endl;
    cali = new CalibStruct[camera_num];
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

    cali = new CalibStruct[camera_num];
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

void Tool::rendering(vector<Mat> &img_set, vector<int> &img_id)
{
    // use two image

    Mat left = img_set[0]; // you need depth image.
    Mat right = img_set[1];

    int left_id = img_id[0];
    int right_id = img_id[1];

    Mat img_target(left.cols,left.rows,CV_8UC3);

    for(int i = 0; i < img_target.cols; ++i)
    {
        for(int j = 0 ; j < img_target.rows; ++j)
        {
            // you need to accelarate the following operation
            // first, get visual view-point image... so , you need to complete function `projFromUVToXYZ` and `projFromXYZToUV`
            // then , follow the paper.
//            if(  )
            ;

        }

    }
}


/**
 *   here need to accelerate
 *
 * project from UV (image coordinate ) to XYZ (3D space)
 *
 *
 * Z_w * x = P * X
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
            int zc = dep.at<cv::Vec3b>(i,j)[0];
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
 *  project from XYZ (3D world coordinate) to UV (image coordinate)
 *
 *  input: cd_
 *  output: rgb, dep
 */

void Tool::projFromXYZToUV(pcl::PointCloud<pcl::PointXYZRGB> &cd_, Eigen::Matrix4d & targetP, Mat &rgb, Mat &dep)
{

    // here you need to initial rgb and depth first since not all the pixel in these two image will be fixed.
    // !!!!

    if(targetP.cols()!=4 || targetP.rows()!=4)
    {
        cerr << " targetP is not a [4x4] matrix!" <<endl;
    }else{
        // initial rgb and depth
        // TODO

        rgb = Mat::zeros(cd_.height,cd_.width,CV_8UC3);
        dep = Mat::zeros(cd_.height,cd_.width,CV_8UC1);

//        imshow("rgb",rgb);
//        imshow("dep",dep);
//        waitKey(0);

        for(int i = 0; i < cd_.height; ++i)
        {
            for(int j = 0 ; j < cd_.width; ++j)
            {
                Vector4d X_;
                X_(0) = cd_.points[i*cd_.width+j].x;
                X_(1) = cd_.points[i*cd_.width+j].y;
                X_(2) = cd_.points[i*cd_.width+j].z;
                X_(3) = 1.0;

                double zc = X_(2);
                Vector4d x_;
                x_ = targetP*X_;

                // TODO

            }
        }

    }

}
