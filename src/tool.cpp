#include "tool.h"

using namespace std;
using namespace fvv_tool;
using namespace cv;

Tool::Tool()
{
    cout << "Tool init." <<endl;
    cali = new ImageFrame[camera_num];
}

Tool::Tool(string dataset_name, int cam_num)
{
    if( dataset_name.compare("ballet") == 0)
    {
        MaxZ = 130;
        MinZ = 42;
    }else{
        MaxZ = 120;
        MinZ = 44;
    }

    camera_num = cam_num;

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
                cout << cali[k].mK[i][j] << "  ";
            }
            cout << endl;
        }
        cout << "camera Extrinsic = " << endl;
        for(int i = 0; i< 3; ++i)
        {
            for(int j = 0; j < 3; ++j)
            {
                cout << cali[k].mR[i][j] << "  ";
            }

            cout << cali[k].mT[i] <<endl;
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
        cerr << "load calibration parameter ..." <<endl;

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

        cout << "load OK." <<endl;

    }

}


/**
 *   load one image( rgb and depth )
 *
 *
 */

void Tool::loadImage(string& campath, vector<int>& camID, int startIndex, int endIndex)
{
    stringstream ss;

    for(int i = 0; i < camID.size(); ++i)
    {
        //color-cam1-f000.jpg, depth-cam1-f000.png

        for(int imgIndex = startIndex; imgIndex < endIndex; ++imgIndex)
        {
            string color_path,dep_path;

            ss << campath << camID[i] << "/color-cam"<<camID[i]<<"-f" << setfill('0')<< setw(3) << imgIndex <<".jpg";
            ss >> color_path;
            ss.clear();

            ss << campath << camID[i] << "/depth-cam"<<camID[i]<<"-f" << setfill('0')<< setw(3) << imgIndex <<".png";
            ss >> dep_path;
            ss.clear();

            cali[camID[i]].rgb = imread(color_path.c_str()); // here can only save one
            cali[camID[i]].dep = imread(dep_path.c_str());   // here can only save one

            color_path.clear();
            dep_path.clear();
        }
    }
    cout << "load Image OK" <<endl;
}


/**
 *
 *   show point cloud
 *
 *    input: pcl::Ptr
 */

void Tool::showPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cd_p)
{
    if(cd_p->size() == 0)
    {
        cout << "cloud point is empty." <<endl;
    }else{
        // although the origin in viewer is not set and the scale is not correct,
        // it not influence the project pixles in target virtual image plane.

        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer1
                (new pcl::visualization::PCLVisualizer("XYZ")); // viewer ID

        viewer1->addPointCloud<pcl::PointXYZ>(cd_p,"XYZ"); // cloud ID
        viewer1->addCoordinateSystem(1.0);
//        viewer1->initCameraParameters();

        viewer1->spin();
    }
}

void Tool::showPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cd_p)
{
    if(cd_p->size() == 0)
    {
        cout << "cloud point is empty." <<endl;
    }else{
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer1
                (new pcl::visualization::PCLVisualizer("XYZRGB")); // viewer ID

        viewer1->addPointCloud<pcl::PointXYZRGB>(cd_p,"XYZRGB"); // cloud ID
        viewer1->addCoordinateSystem(1);
//        viewer1->setCameraPosition();
//        viewer1->initCameraParameters();

        viewer1->spin();
    }
}


/**
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

    cout << "generate each camera's P, OK" <<endl;

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

void Tool::rendering(vector<int> &img_id, Matrix4d& targetP )
{
    // use two image

    Mat left_d = cali[img_id[0]].dep;
    Mat right_d = cali[img_id[1]].dep;

    Mat vir_depth = Mat::zeros(left_d.rows,left_d.cols,CV_8UC1); // depth image in novel viewpoint
    Mat vir_rgb = Mat::zeros(left_d.rows,left_d.cols,CV_8UC3);   // rgb image in novel viewpoint

    Mat left_r = cali[img_id[0]].rgb;
    Mat right_r = cali[img_id[1]].rgb;

    Matrix<double,3,1> left_T;
    Matrix<double,3,1> right_T;
    Matrix<double,3,1> target_T;

    left_T = (cali[img_id[0]].mP).block(0,3,3,1);
    right_T = (cali[img_id[1]].mP).block(0,3,3,1);
    target_T = targetP.block(0,3,3,1);


    // point cloud's point is link to image's pixel , which has the same index.!!!
    // which makes `vir_link_ori` more easier.
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_l_cd( new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_r_cd( new pcl::PointCloud<pcl::PointXYZ>);


    /**
     * project left depth image to virtual image plane
     * project right depth image to virtual image plane
     * fuseing these two depth images and get one.
     *
     **/

    Mat left_vir_d = Mat::zeros(left_d.rows, left_d.cols,CV_8UC1); // left project to virtual depth image.
    std::vector<cv::Point2i> left_vir_link_orig; // used to link pixels in origin image to those pixels in virtual image plane

    projFromUVToXYZ(left_d,img_id[0],tmp_l_cd);
    projFromXYZToUV(tmp_l_cd,targetP,left_vir_d,left_vir_link_orig);

    Mat right_vir_d = Mat::zeros(left_d.rows, left_d.cols,CV_8UC1); // right project to virtual depth image
    std::vector<cv::Point2i> right_vir_link_orig; // used to link pixels in origin image to those pixels in virtual image plane

    projFromUVToXYZ(right_d,img_id[1],tmp_r_cd);
    projFromXYZToUV(tmp_r_cd,targetP,right_vir_d,right_vir_link_orig);

    fusingDepth(left_vir_d,right_vir_d,vir_depth);
    cout << "fusing depth over." <<endl;

//    imshow("vir",vir_depth);
//    waitKey(0);

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
    fusingRgb(left_r,left_d,left_vir_link_orig, left_T,
              right_r,right_d,right_vir_link_orig, right_T,
              vir_rgb, target_T );

    addWeighted(vir_rgb,0.5,cali[4].rgb,0.5,0,vir_rgb);
//    imshow("target",cali[4].rgb);

    imshow("vir_rgb",vir_rgb);
    waitKey(0);
    cout << "fusing rgb over." <<endl;

}

/**
 * generate rgb image in novel viewpoint image plane
 *
 * input: left and right rgb image
 * output: virtual imahe plane rgb image
 */

void Tool::fusingRgb(Mat &left_rgb, Mat &left_dep, vector<Point2i> &left_vir_link_orig, Matrix<double,3,1>& left_T,
                     Mat &right_rgb, Mat &right_dep, vector<Point2i> &right_vir_link_orig, Matrix<double,3,1>& right_T,
                     Mat &target_rgb, Matrix<double,3,1>& target_T)
{
    Array3d t1 = target_T - left_T;
    Array3d t2 = target_T - right_T;

    double sub_L = sqrt(t1.square().sum());
    double sub_R = sqrt(t2.square().sum());

    double alpha = (sub_L)/(sub_L + sub_R);

//    target_rgb = cv::Mat::zeros(left_rgb.rows,left_rgb.cols,CV_8UC3);

    for(int i = 0; i < left_rgb.rows; ++i)
    {
        for(int j = 0; j < left_rgb.cols; ++j)
        {

            // for those u,v that don not match any ul,vl/ur,vr , since their depth value has been set 0,
            // which must smaller than THRESOLD, so, Point(-1,-1) will not be used to find pixel that
            // in left image or right image.

            // calculate ZR(u,v) and ZL(u,v), Z is depth
            int occL = left_dep.at<uchar>(i,j) > THRESHOLD ? 0 : 1;
            int occR = right_dep.at<uchar>(i,j) > THRESHOLD ? 0 : 1;

            if(occL == 0 && occR == 0)
            {
                target_rgb.at<cv::Vec3b>(i,j)[0] = (1-alpha)*left_rgb.at<cv::Vec3b>(left_vir_link_orig[i*left_rgb.cols+j])[0]
                        + alpha*right_rgb.at<cv::Vec3b>(right_vir_link_orig[i*left_rgb.cols+j])[0];
                target_rgb.at<cv::Vec3b>(i,j)[1] = (1-alpha)*left_rgb.at<cv::Vec3b>(left_vir_link_orig[i*left_rgb.cols+j])[1]
                        + alpha*right_rgb.at<cv::Vec3b>(right_vir_link_orig[i*left_rgb.cols+j])[1];
                target_rgb.at<cv::Vec3b>(i,j)[2] = (1-alpha)*left_rgb.at<cv::Vec3b>(left_vir_link_orig[i*left_rgb.cols+j])[2]
                        + alpha*right_rgb.at<cv::Vec3b>(right_vir_link_orig[i*left_rgb.cols+j])[2];

            }else if(occL == 0 && occR == 1)
            {
                target_rgb.at<cv::Vec3b>(i,j)[0] = left_rgb.at<cv::Vec3b>(left_vir_link_orig[i*left_rgb.cols+j])[0];
                target_rgb.at<cv::Vec3b>(i,j)[1] = left_rgb.at<cv::Vec3b>(left_vir_link_orig[i*left_rgb.cols+j])[1];
                target_rgb.at<cv::Vec3b>(i,j)[2] = left_rgb.at<cv::Vec3b>(left_vir_link_orig[i*left_rgb.cols+j])[2];

            }else if (occL == 1 && occR == 0)
            {
                target_rgb.at<cv::Vec3b>(i,j)[0] = right_rgb.at<cv::Vec3b>(right_vir_link_orig[i*left_rgb.cols+j])[0];
                target_rgb.at<cv::Vec3b>(i,j)[1] = right_rgb.at<cv::Vec3b>(right_vir_link_orig[i*left_rgb.cols+j])[1];
                target_rgb.at<cv::Vec3b>(i,j)[2] = right_rgb.at<cv::Vec3b>(right_vir_link_orig[i*left_rgb.cols+j])[2];

            }else{
                ;
            }

        }
    }
}

/**
 *  fusing two depth image
 *
 */

void Tool::fusingDepth(Mat &left_, Mat &right_, Mat &target)
{
    if(left_.cols!=right_.cols || left_.rows!= right_.rows)
    {
        cerr << "Error! in function [fusingDepth], input left image and right image not in the same size." <<endl;
    }else{

        cout << "in fusing." <<endl;

//        target = Mat::zeros(left.rows,left.cols,CV_8UC1); // fuck ??!!!

        int count = 0;
        for(int i = 0; i < left_.rows; ++i)
        {
            for(int j = 0; j < left_.cols; ++j)
            {

                if (left_.at<uchar>(i,j) < 1 && right_.at<uchar>(i,j) < 1)
                { // both not have value

                    count = count + 1; // count those empty point
                    continue; // stay 0
                }else if ( left_.at<uchar>(i,j) < 1 && right_.at<uchar>(i,j) >=1 )
                {
//                    cout << "                    R"<<endl;
                    target.at<uchar>(i,j) = right_.at<uchar>(i,j);

                }else if (left_.at<uchar>(i,j) >= 1 && right_.at<uchar>(i,j) < 1)
                {
//                    cout << "L                    "<<endl;
                    target.at<uchar>(i,j) = left_.at<uchar>(i,j);

                }else{
                    // both has value
//                    cout << "          =          "<<endl;
                    if(left_.at<uchar>(i,j) > right_.at<uchar>(i,j) )
                    {
                        target.at<uchar>(i,j) = right_.at<uchar>(i,j);
                    }else{
                        target.at<uchar>(i,j) = left_.at<uchar>(i,j);
                    }
                }
            }

        }

        cerr << endl << "In fusing Depth image, " << (count)/(left_.rows * left_.cols)
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
// this input dep is Vec3b ! not uchar
void Tool::projFromUVToXYZ(Mat &rgb, Mat &dep, int index, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cd_)
{
    cd_->width = rgb.cols;
    cd_->height = rgb.rows;
    cd_->points.resize(cd_->width * cd_->height);

    for(int i = 0; i < cd_->height; ++i)
    {
        for(int j = 0; j < cd_->width; ++j)
        {
            double zc = getPixelActualDepth(dep.at<cv::Vec3b>(i,j)[0]); // actual depth
            int u = j;
            int v = i;

            Matrix4d p_in = (cali[index].mP).inverse();
            Vector4d x_(zc*u,zc*v,zc,1);
            Vector4d X_;
            X_ = p_in*x_;

            cd_->points[i*cd_->width+j].x = X_(0);
            cd_->points[i*cd_->width+j].y = X_(1);
            cd_->points[i*cd_->width+j].z = X_(2);

            cd_->points[i*cd_->width+j].r = rgb.at<cv::Vec3b>(i,j)[2];
            cd_->points[i*cd_->width+j].g = rgb.at<cv::Vec3b>(i,j)[1];
            cd_->points[i*cd_->width+j].b = rgb.at<cv::Vec3b>(i,j)[0];

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

void Tool::projFromUVToXYZ( Mat &dep, int index, pcl::PointCloud<pcl::PointXYZ>::Ptr cd_)
{
    cd_->width = dep.cols;
    cd_->height = dep.rows;
    cd_->points.resize(cd_->width * cd_->height);

    for(int i = 0; i < cd_->height; ++i)
    {
        for(int j = 0; j < cd_->width; ++j)
        {
            double zc = getPixelActualDepth(dep.at<cv::Vec3b>(i,j)[0]); // actual depth
            int u = j;
            int v = i;

            Matrix4d p_in = (cali[index].mP).inverse();
            Vector4d x_(zc*u,zc*v,zc,1);
            Vector4d X_;
            X_ = p_in*x_;

            cd_->points[i*cd_->width+j].x = X_(0);
            cd_->points[i*cd_->width+j].y = X_(1);
            cd_->points[i*cd_->width+j].z = X_(2);

        }

    }

}

/**
 *  project from XYZ (3D world coordinate) to UV (image coordinate)
 *
 *  input: cd_
 *  output: rgb, dep
 */

void Tool::projFromXYZToUV(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cd_,
                           Eigen::Matrix4d & targetP,
                           Mat &rgb, Mat &dep,
                           std::vector<cv::Point>& vir_link_ori)
{
    // here you need to initial rgb and depth first since not all the pixel in these two image will be fixed.

    if(targetP.cols()!=4 || targetP.rows()!=4)
    {
        cerr << " targetP is not a [4x4] matrix!" <<endl;
    }else{
        // initial rgb and depth
        // TODO

        cout << "XYZRGB project UV .." <<endl;

        rgb = Mat::zeros(cd_->height,cd_->width,CV_8UC3);
        dep = Mat::zeros(cd_->height,cd_->width,CV_8UC1);

        for(int i = 0; i < cd_->height; ++i)
        {
            for(int j = 0 ; j < cd_->width; ++j)
            {
                Vector4d X_;
                X_(0) = cd_->points[i*cd_->width+j].x;
                X_(1) = cd_->points[i*cd_->width+j].y;
                X_(2) = cd_->points[i*cd_->width+j].z; // actual depth
                X_(3) = 1.0;

                double zc = X_(2);
                Vector4d x_;
                x_ = targetP*X_;

                if(zc < 0.2) // important in test.
                {
                    vir_link_ori.push_back(Point(-1,-1));
                    continue;
                }
                if(x_(0) < 0 || x_(1) < 0)
                {
                    vir_link_ori.push_back(Point(-1,-1));
                    continue;
                }

                rgb.at<cv::Vec3b>(int(x_(1)/zc),int(x_(0)/zc))[0] = cd_->points[i*cd_->width+j].b;
                rgb.at<cv::Vec3b>(int(x_(1)/zc),int(x_(0)/zc))[1] = cd_->points[i*cd_->width+j].g;
                rgb.at<cv::Vec3b>(int(x_(1)/zc),int(x_(0)/zc))[2] = cd_->points[i*cd_->width+j].r;

                dep.at<uchar>(int(x_(1)/zc),int(x_(0)/zc)) = getPixelDepth(zc);
            }
        }
        cout << "XYZRGB project to UV .. OK" <<endl;
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

void Tool::projFromXYZToUV(pcl::PointCloud<pcl::PointXYZ>::Ptr cd_,
                           Eigen::Matrix4d & targetP,
                           Mat &dep,
                           std::vector<cv::Point>& vir_link_ori)
{

    // here you need to initial rgb and depth first since not all the pixel in these two image will be fixed.

    if(targetP.cols()!=4 || targetP.rows()!=4)
    {
        cerr << " targetP is not a [4x4] matrix!" <<endl;
    }else{
        // initial depth
        // TODO

        // here, you should not clear it if it is empty.
        if(!vir_link_ori.empty())
        {
            vir_link_ori.clear();
        }

        cout << "Start project XYZ to UV.." <<endl;
        for(int i = 0; i < cd_->height; ++i)
        {
            for(int j = 0 ; j < cd_->width; ++j)
            {
                Vector4d X_;
                X_(0) = cd_->points[i*cd_->width+j].x;
                X_(1) = cd_->points[i*cd_->width+j].y;
                X_(2) = cd_->points[i*cd_->width+j].z; // actual depth
                X_(3) = 1.0;

                double zc = X_(2);
                Vector4d x_;
                x_ = targetP*X_;


                // these judge operate is every important !!!!
                // since the program will not stop if you locate a wide-point.
                // especially the third one.
                if(zc < 0.2) // important in test.
                {
                    vir_link_ori.push_back(Point(-1,-1));
                    continue;
                }

                if(x_(0) < 0 || x_(1) < 0)
                {
                    vir_link_ori.push_back(Point(-1,-1));
                    continue;
                }

                if( int(x_(1)/zc) >= dep.rows || int(x_(0)/zc) >= dep.cols)
                {
                    vir_link_ori.push_back(Point(-1,-1));
                    continue;
                }

                dep.at<uchar>(int(x_(1)/zc),int(x_(0)/zc)) = getPixelDepth(zc);

                vir_link_ori.push_back(Point(int(x_(0)/zc),int(x_(1)/zc)));
            }
        }

        cout << "XYZ project to UV .. OK" << endl;
    }

}
