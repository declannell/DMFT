if [[ -d "textfiles" ]]
then
    echo "textfiles exists on your filesystem."
else
  mkdir textfiles
fi

cmake -S . -B target/
cmake --build target
