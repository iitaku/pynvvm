#ifndef PYLLVM_CONTEXT_H
#define PYLLVM_CONTEXT_H

#include <llvm/LLVMContext.h>

namespace pyllvm {

class PyLLVMContext
{
  public:  
    PyLLVMContext(llvm::LLVMContext& obj)
      : obj_(obj) {}

    llvm::LLVMContext& obj_;
};

PyLLVMContext *get_global_context();

} /* namespace pyllvm */

#endif /* PYLLVM_CONTEXT_H */
