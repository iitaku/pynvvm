#ifndef PYLLVM_CONSTANTS_H
#define PYLLVM_CONSTANTS_H

#include <llvm/Constants.h>

#include "pyllvm-value.h"

namespace pyllvm {

class PyLLVMType;

class PyLLVMConstantInt : public PyLLVMValue
{
  public:
    PyLLVMConstantInt(llvm::Constant *obj)
      : PyLLVMValue(obj) {}

    static
    PyLLVMConstantInt *get(PyLLVMType *type, int i);
};

} /* namespace pyllvm */

#endif /* PYLLVM_CONSTANTS_H */

