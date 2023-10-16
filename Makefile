all:
	nvcc source.cu -o source

clean:
	rm -f source