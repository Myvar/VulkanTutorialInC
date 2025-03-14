cd "$(dirname "$0")"
set -x
mkdir -p build_linux
cd build_linux
cmake -G "Unix Makefiles" \
  ../..
rm vkt
make -j

cp compile_commands.json ../..
