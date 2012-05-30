#ifndef PYLLVM_TYPE_H
#define PYLLVM_TYPE_H

#include <llvm/Type.h>

namespace pyllvm {

class PyLLVMContext;

class PyLLVMType
{
  public:
    PyLLVMType(llvm::Type* obj)
      : obj_(obj) {}

    static
    PyLLVMType *get_int32_ty(PyLLVMContext *context);

    llvm::Type *obj_;
};

} /* namespace pyllvm */

#endif /* PYLLVMFUNCTION_TYPE_H */
