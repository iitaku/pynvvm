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
    PyLLVMConstantInt *get(PyLLVMType *type, int val);
};

class PyLLVMConstantFP : public PyLLVMValue
{
  public:
    PyLLVMConstantFP(llvm::Constant *obj)
      : PyLLVMValue(obj) {}

    static
    PyLLVMConstantFP *get(PyLLVMType *type, double val);
};

} /* namespace pyllvm */

#endif /* PYLLVM_CONSTANTS_H */

