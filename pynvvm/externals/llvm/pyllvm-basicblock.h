#ifndef PYLLVM_BASICBLOCK_H
#define PYLLVM_BASICBLOCK_H

#include <llvm/BasicBlock.h>

namespace pyllvm {

class PyLLVMContext;
class PyLLVMFunction;

class PyLLVMBasicBlock
{
  public:
    PyLLVMBasicBlock(llvm::BasicBlock *obj)
      : obj_(obj) {}

    static
    PyLLVMBasicBlock *create(PyLLVMContext *context, std::string id, PyLLVMFunction *function);
    
    llvm::BasicBlock *obj_;
};

} /* namespace pyllvm */

#endif /* PYLLVM_BASICBLOCK_H */

