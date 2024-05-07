BUILD_DIR=`pwd`/../build
export OMP_NUM_THREADS=32
REPEAT=10
TILE=32
SIZE=256
DTYPE=f32
VNNI=1

cd $BUILD_DIR
cmake --build . 

CORRECTNESS_WORKLOAD="python ../scripts/correctness_check.py --M=${SIZE} --N=${SIZE} --K=${SIZE}"
PERF_WORKLOAD="./bin/mlir-gen --batch=${SIZE} --layers=${SIZE},${SIZE} --tiles=${TILE},${TILE},${TILE} --float-type=${DTYPE} --vnni=${VNNI}"

WORKLOAD=${PERF_WORKLOAD}

${WORKLOAD} |  numactl -N 1 --membind=1 ./bin/tpp-run  -e entry -entry-point-result=void -n $REPEAT --print-mlir=late --def-parallel

