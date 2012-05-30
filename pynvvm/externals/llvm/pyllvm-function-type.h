#ifndef PYLLVM_FUNCTION_TYPE_H
#define PYLLVM_FUNCTION_TYPE_H

#include <llvm/DerivedTypes.h>

#include "pyllvm.h"

namespace pyllvm {

class PyLLVMType;

class PyLLVMFunctionType
{
  public:
    PyLLVMFunctionType(llvm::FunctionType* obj)
      : obj_(obj) {}

    static
    PyLLVMFunctionType *get(PyLLVMType *type, bool is_var_arg);

    llvm::FunctionType *obj_;
};

} /* namespace pyllvm */

#endif /* PYLLVM_FUNCTION_TYPE_H */
