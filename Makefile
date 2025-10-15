all:
	nvcc -arch=sm_75 -o test test.cu -lcublas

clean:
	rm -f test