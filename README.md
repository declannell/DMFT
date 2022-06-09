Run the `build.sh` script with `./build.sh`. This reads runs `cmake`, which finds the required `.cpp` files in the `src/` directory. CMake works magic and then creates a target directory. In this target directory, a Makefile is create and run. DO NOT EDIT THIS MAKEFILE.

The Makefile creates an executable which can be run by `target/dmft`. The executable creates several text files and stores them in my `textfile` directory. (I should make this directory appear in the current directory for portability).

One can change the parameters for which the code runs by editing the file `parameters.cpp`, found within `src/` directory.
