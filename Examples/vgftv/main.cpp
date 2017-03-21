#include "tool.h"

using namespace std;
using namespace fvv_tool;
using namespace Eigen;



int main(int argc, char ** argv)
{
    Tool tool;
    tool.loadImageParameter("./../../dataset/MSR3DVideo-Breakdancers/calibParams-breakdancers.txt");

//    tool.showParameter();

    tool.generateP();

//    cout << tool.cali[0].mP <<endl;

    // now each tool.cali[] has mP, cali[4]

    /**
     * depth image operation, post-filtering
     *
     *  TODO
     *
     * **/



    /**
     *  project the nearest image to viewpoint
     *
     *  TODO
     * */




    return 0;
}
