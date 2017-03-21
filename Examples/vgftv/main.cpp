#include "tool.h"

using namespace std;
using namespace fvv_tool;
using namespace Eigen;

int main(int argc, char ** argv)
{
    Tool tool;
    tool.loadImageParameter("./../../dataset/MSR3DVideo-Breakdancers/calibParams-breakdancers.txt");

    tool.showParameter();


    return 0;
}
