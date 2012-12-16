OBJS=pyllvm.o               \
     pyllvm-argument.o      \
     pyllvm-basicblock.o    \
     pyllvm-builder.o       \
     pyllvm-constants.o     \
     pyllvm-context.o       \
     pyllvm-function.o      \
     pyllvm-metadata.o      \
     pyllvm-module.o        \
     pyllvm-type.o          \
     pyllvm-value.o 

all: pyllvm.so

clean:
	rm -rf *.o *.so

pyllvm.so: $(OBJS)
	clang++ $(OBJS) -o $@ -fPIC --shared `llvm-config --ldflags` -lboost_python-mt -lpython2.7 `llvm-config --libs`

.cc.o:
	clang++ $< -fPIC -c -I /usr/include/python2.7 `llvm-config --cppflags`
