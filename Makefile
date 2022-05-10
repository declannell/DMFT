#CC = g++

#CFLAGS = -g -Wall -O3 -ffast-math #-D_CC_OVERLAP 

#LDFLAGS = -lm

#GFOBJS = parameters.o leads_self_energy.o interacting_gf.o

#EXECS = a

##i think the problem is that i dont have explicitly the dmft depends on parameters.

#all: $(EXECS)

#a: main.o  $(GFOBJS) 
#	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

#tags:
#	etags *.c *.h

#.PHONY: clean tags tests

#clean:
#	$(RM) *.o $(EXECS) $(TESTS) TAGS tags

all:
	g++  -O3 -g -c -o main.o main.cpp
	g++  -O3 -g -c -o parameters.o parameters.cpp
	g++  -O3 -g -c -o leads_self_energy.o leads_self_energy.cpp
	g++  -O3 -g -c -o interacting_gf.o interacting_gf.cpp
	g++  -O3 -g -c -o dmft.o dmft.cpp
	g++ -g -Wall -O3 -o a main.o parameters.o leads_self_energy.o interacting_gf.o dmft.o -lm


tags:
	etags *.c *.h

.PHONY: clean tags tests

clean:
	$(RM) *.o $(EXECS) $(TESTS) TAGS tags
