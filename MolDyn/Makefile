# Makefile for the molecular dynamics code

#
# C compiler and options for Intel
#
#CC=     icc -O3 -qopenmp -std=c99
#LIB=    -lm

#
# C compiler and options for GNU 
#
#
#CC=		gcc -03 -fopenmp
CC =  clang -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp
LIB=	-lm

#
# Object files
#
OBJ=    main.o \
	dfill.o \
	domove.o \
        dscal.o \
	fcc.o \
	forces.o \
	mkekin.o \
	mxwell.o \
	prnout.o \
	velavg.o

#
# Compile
#
md:	$(OBJ)
	$(CC) -o $@ $(OBJ) $(LIB)

.c.o:
	$(CC) -c $<

#
# Clean out object files and the executable.
#
clean:
	rm *.o md
