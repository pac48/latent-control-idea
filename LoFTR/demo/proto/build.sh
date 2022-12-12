# rm ./proto/*.py
protoc -I=./src --python_out=. ./src/msgs.proto
