#include "test_lib.h"

using namespace std;

namespace test{

	TEST::TEST()
	{
		cout << "test initilate" <<endl;
		class_name = "TEST";
	}

	TEST::~TEST()
	{
		cout << "test destory" <<endl;
	}

	void TEST::show()
	{
		cout <<"class name: " << class_name <<endl;
        cout <<"test git qtcreator" <<endl;
	}

}
