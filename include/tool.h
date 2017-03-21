#ifndef TOOL_H
#define TOOL_H

#include "stdio.h"
#include "iostream"
#include "fstream"
#include "vector"
#include "string"
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

namespace fvv_tool
{

        struct CalibStruct{
		double mK[3][3];
		double mR[3][3];
		double mT[3];

                Matrix4d mP;
                // mP=[mK*mR mK*mT ]
                //    [ 0      1   ]

                // Z*x=mP*X
        };

class Tool
{
public:
    Tool();
    Tool(string dataset_name);
    ~Tool();

    // operate *cali ;
    void loadImageParameter(char* file_name);

    void showParameter();

    // generate mP
    void generateP();

    CalibStruct* cali;

private:
    int camera_num = 8;
    int MaxZ = 120;
    int MinZ = 44;

};


}
#endif // TOOL_H
