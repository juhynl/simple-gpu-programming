CC = nvcc

TARGET_MY_PROGRAM = matmul
SRC_MY_PROGRAM = matrix_multiplication.cu

TARGET_MATGEN = matgen
SRC_MATGEN = matrix_generator.cu

FLAGS = -lcublas -arch=sm_86 -Xcompiler -fopenmp

.PHONY: all clean

all: $(TARGET_MY_PROGRAM) $(TARGET_MATGEN)

$(TARGET_MY_PROGRAM): $(SRC_MY_PROGRAM)
	$(CC) $(SRC_MY_PROGRAM) -o $(TARGET_MY_PROGRAM) $(FLAGS)

$(TARGET_MATGEN): $(SRC_MATGEN)
	$(CC) $(SRC_MATGEN) -o $(TARGET_MATGEN) $(FLAGS)

clean:
	rm -f $(TARGET_MY_PROGRAM) $(TARGET_MATGEN)
