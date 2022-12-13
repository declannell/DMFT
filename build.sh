if [[ -d "textfiles" ]]
then
    echo "textfiles exists on your filesystem."
else
  mkdir textfiles
fi
module load intel/19.0.5.281   intel-mpi/2019.5.281    intel-mkl/2019.5.281
cmake -S . -B target/
cmake --build target
