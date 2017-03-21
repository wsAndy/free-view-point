#include "tool.h"

using namespace std;
using namespace fvv_tool;

Tool::Tool()
{
    cout << "Tool init." <<endl;
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
        float tmp; // dummy
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

            fscanf(f_read,"%lf",tmp);
            fscanf(f_read,"%lf",tmp);

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


}
