CC=nvcc
SRC=montecarlo_simulation.cu
TARGET=mc_sim
FLAGS=-lcublas -arch=sm_86 -Xcompiler -fopenmp

all : $(TARGET)

$(TARGET) : $(SRC)
	$(CC) $(SRC) -o $(TARGET) $(FLAGS)

.PHONY : clean
clean :
	rm $(TARGET)