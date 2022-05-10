CC = g++

CFLAGS = -g -Wall #-D_CC_OVERLAP

LDFLAGS = -lm

GFOBJS = parameters.o leads_self_energy.o interacting_gf.o

EXECS = a

##i think the problem is that i dont have explicitly the dmft depends on parameters.

all: $(EXECS)

a: main.o  $(GFOBJS) 
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

tags:
	etags *.c *.h

.PHONY: clean tags tests

clean:
	$(RM) *.o $(EXECS) $(TESTS) TAGS tags






#CC = g++
#
#CFLAGS = -g -Wall #-D_CC_OVERLAP
#
#LDFLAGS = -lm
#
#POISSOBJS = parameters.o leads_self_energy.o

#
#EXECS = a
#
#
#
#all: $(EXECS)
#
#a: main.o  $(POISSOBJS) 
#	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
#
#
#
## tmpedc1d: testdc1d.o decomp1d.o
#
### tests
#
#tags:
#	etags *.c *.h
#
#.PHONY: clean tags tests
#
#clean:
#	$(RM) *.o $(EXECS) $(TESTS) TAGS tags
#