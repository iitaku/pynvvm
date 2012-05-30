OBJS=pyllvm.o               \
     pyllvm-builder.o       \
     pyllvm-context.o       \
     pyllvm-function.o      \
     pyllvm-function-type.o \
     pyllvm-module.o        \
     pyllvm-type.o 

all: pyllvm.so

clean:
	rm -rf *.o *.so

pyllvm.so: $(OBJS)
	g++ $(OBJS) -o $@ -fPIC --shared `llvm-config --cppflags --ldflags --libs core bitWriter` -lboost_python -lpython2.7 -Wl,-soname,pyllvm.so

.cc.o:
	g++ $< -fPIC -c -I /usr/include/python2.7 `llvm-config --cppflags`
