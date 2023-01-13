Run the `build.sh` script with `./build.sh`. This reads runs `cmake`, which finds the required `.cpp` files in the `src/` directory. CMake works magic and then creates a target directory. In this target directory, a Makefile is create and run. DO NOT EDIT THIS MAKEFILE.

The Makefile creates an executable which can be found within `target/dmft`. The executable creates several text files and stores them in the current directory. 

To run the code, all you need is the input_file and relative path to the executable. One can change the parameters for which the code runs by editing the input_file . The out file contains lots of info about the results. The code is run by `mpirun -np 1 target/dmft`
