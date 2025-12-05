NVCC = nvcc
CC   = g++
FLAGS= -DDEBUG
LIBS = -lm
ALWAYS_REBUILD = makefile

nbody: nbody.o compute.o
	$(NVCC) $(FLAGS) $^ -o $@ $(LIBS)

nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	$(CC) $(FLAGS) -x c++ -c $<

compute.o: compute.cu config.h vector.h $(ALWAYS_REBUILD)
	$(NVCC) $(FLAGS) -c $<

clean:
	rm -f *.o nbody
