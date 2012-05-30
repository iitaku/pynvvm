#ifndef PYLLVM_BUILDER_H
#define PYLLVM_BUILDER_H

#include <llvm/Support/IRBuilder.h>

namespace pyllvm {

class PyLLVMContext;

class PyLLVMBuilder
{
  public:
    PyLLVMBuilder(llvm::IRBuilder<>* obj)
      : obj_(obj) {}
 
    llvm::IRBuilder<> *obj_;
};

PyLLVMBuilder *create_builder(PyLLVMContext *context);

} /* namespace pyllvm */

#endif /* PYLLVM_BUILDER_H */
