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
	g++ $(OBJS) -o $@ -fPIC --shared -L /opt/local/lib `llvm-config --cppflags --ldflags --libs core bitWriter` -lboost_python -lpython2.7

.cc.o:
	g++ $< -fPIC -c -I /opt/local/include -I /usr/include/python2.7 `llvm-config --cppflags`
