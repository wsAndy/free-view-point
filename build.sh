
rm -rf ./build

echo "building..."

mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j8

