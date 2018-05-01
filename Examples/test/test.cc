#include "test_lib.h"

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include <iostream>

using namespace std;
using namespace  cv;

void showMatch(Mat& img1, vector<KeyPoint>& key1, Mat& img2, vector<KeyPoint>& key2, vector<DMatch>& match, Mat& output);

int main()
{
//	test::TEST tt;

//	tt.show();


    //读取原始基准图和待匹配图
     Mat rgbd1 = imread("/Users/sheng/CarVideo/kitti/04/image_2/000180.png");
     Mat rgbd2 = imread("/Users/sheng/CarVideo/kitti/04/image_3/000180.png");

//     cv::resize(rgbd1, rgbd1, cv::Size(rgbd1.cols/2, rgbd1.rows/2));
//     cv::resize(rgbd2, rgbd2, cv::Size(rgbd2.cols/2, rgbd2.rows/2));

        Ptr<ORB> orb = ORB::create();
        vector<KeyPoint> Keypoints1,Keypoints2;
        Mat descriptors1,descriptors2;
        orb->detectAndCompute(rgbd1, Mat(), Keypoints1, descriptors1);
        orb->detectAndCompute(rgbd2, Mat(), Keypoints2, descriptors2);

        //cout << "Key points of image" << Keypoints.size() << endl;

        //可视化，显示关键点
        Mat ShowKeypoints1, ShowKeypoints2;
        drawKeypoints(rgbd1,Keypoints1,ShowKeypoints1);
        drawKeypoints(rgbd2, Keypoints2, ShowKeypoints2);

        //Matching
        vector<DMatch> matches;
        Ptr<DescriptorMatcher> matcher =DescriptorMatcher::create("BruteForce");
        matcher->match(descriptors1, descriptors2, matches);
        cout << "find out total " << matches.size() << " matches" << endl;

        //可视化
        Mat ShowMatches;
        showMatch(rgbd1,Keypoints1,rgbd2,Keypoints2,matches,ShowMatches);
        imshow("matches_no_ransac", ShowMatches);
        cv::imwrite("/Users/sheng/Desktop/180_185_2.jpg",ShowMatches);




        // 判断距离,首先是垂直方向不能超过两个像素
        vector<DMatch> goodmatch,verygood;
//        for(int i = 0; i < matches.size(); ++i)
//        {
//            if( fabs( Keypoints1[matches[i].queryIdx].pt.y - Keypoints2[matches[i].trainIdx].pt.y  ) <= 2 )
//            {
//                goodmatch.push_back(matches[i]);

//            }
//        }
        goodmatch = matches;

        cout << "goodmatch="<< goodmatch.size() << endl;


        double min_dis = 1000;
        for(int i = 0; i < goodmatch.size(); ++i)
        {
            if(min_dis > goodmatch[i].distance)
            {
                min_dis = goodmatch[i].distance;
            }
        }

        int factor = 7;

        for(int i = 0; i < goodmatch.size(); ++i)
        {
            if( goodmatch[i].distance < factor*min_dis )
            {
                verygood.push_back(goodmatch[i]);
            }
        }
        cout << "goodmatch later="<< verygood.size() << endl;

//        Mat ShowMatches2;
//        showMatch(rgbd1,Keypoints1,rgbd2,Keypoints2,verygood,ShowMatches2);
//        imshow("matches_no_ransac_fix", ShowMatches2);
//        cv::imwrite("/Users/sheng/Desktop/180_185_2.jpg",ShowMatches2);




        vector<Point2f> goodleft, goodright;
        vector<KeyPoint> leftkey1, rightkey2;
        for(int i = 0; i < verygood.size(); ++i)
        {
            leftkey1.push_back( Keypoints1[verygood[i].queryIdx] );
            rightkey2.push_back( Keypoints2[verygood[i].trainIdx] );
        }

        for(int i = 0; i < leftkey1.size(); ++i)
        {
            goodleft.push_back(Point2f( leftkey1[i].pt ));
            goodright.push_back(Point2f( rightkey2[i].pt ));
        }

        vector<uchar> status;
        findFundamentalMat(goodleft,goodright,CV_FM_RANSAC,2,0.999999,status);

        cout << status.size() << endl;

        vector<KeyPoint> ll1,rr2;

        vector<DMatch> mmm;
        int ind = 0;
        for(int i = 0; i < verygood.size(); ++i)
        {
            if(status[i] != 0)
            {
                ll1.push_back(leftkey1[i]);
                rr2.push_back(rightkey2[i]);
                verygood[i].queryIdx = ind;
                verygood[i].trainIdx = ind;
                mmm.push_back(verygood[i]);
                ind = ind + 1;
            }
        }
        cout << "=========" <<endl;

        Mat ShowMatches3;
        showMatch(rgbd1,ll1,rgbd2,rr2,mmm,ShowMatches3);
        imshow("matches_ransac_fix", ShowMatches3);
        cv::imwrite("/Users/sheng/Desktop/180_185_2_ransac.jpg",ShowMatches3);

        waitKey(0);

}

void showMatch(Mat& img1, vector<KeyPoint>& key1, Mat& img2, vector<KeyPoint>& key2, vector<DMatch>& match, Mat& output)
{
    int row = img1.rows;
    int col = img1.cols;

    output = Mat::zeros(row*2,col,CV_8UC3);

    for(int i = 0; i < row; ++i)
    {
        for(int j = 0; j < col; ++j)
        {
            output.at<Vec3b>(i,j)[0] = img1.at<Vec3b>(i,j)[0];
            output.at<Vec3b>(i,j)[1] = img1.at<Vec3b>(i,j)[1];
            output.at<Vec3b>(i,j)[2] = img1.at<Vec3b>(i,j)[2];
        }
    }
    for(int i = row; i < row * 2; ++i)
    {
        for(int j = 0; j < col; ++j)
        {
            output.at<Vec3b>(i,j)[0] = img2.at<Vec3b>(i-row,j)[0];
            output.at<Vec3b>(i,j)[1] = img2.at<Vec3b>(i-row,j)[1];
            output.at<Vec3b>(i,j)[2] = img2.at<Vec3b>(i-row,j)[2];
        }
    }


    for(int i = 0; i < match.size(); ++i)
    {
        cout << match[i].distance << endl;
//        if(match[i].distance < )
        {
            int i1 = match[i].queryIdx;
            int i2 = match[i].trainIdx;
            KeyPoint kp1 = key1[i1], kp2 = key2[i2];

            Point2f pt1 = kp1.pt, pt2 = kp2.pt;
            Point2f dpt2 = Point2f( pt2.x, pt2.y + row );

            cv::circle(output,pt1,2,Scalar(0,255,0),1);
            cv::circle(output,dpt2,2,Scalar(0,255,0),1);

            cv::line(output,pt1,dpt2,Scalar(0,0,255),1);

        }
    }




}

