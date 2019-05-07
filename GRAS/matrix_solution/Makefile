CC:=g++
LD:=ld
AR:=ar
SH:=sh

CFLAGS:= -D_REENTERANT -DLCOMP_LINUX=1 -rdynamic -O0 -Wall -I$(TOPDIR)/include -L$(TOPDIR)/lib


CFLAGS+= -I../include -I/usr/include/nptl -L/usr/lib/nptl -ldl -lpthread

   
test: DAC.cpp
	$(CC) DAC.cpp $(CFLAGS) -o DAC

clean:
	rm DAC
