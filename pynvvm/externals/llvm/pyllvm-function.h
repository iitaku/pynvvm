#ifndef PYLLVM_FUNCTION_H
#define PYLLVM_FUNCTION_H

#include <string>

#include <llvm/Function.h>

#include "pyllvm.h"

namespace pyllvm {

class PyLLVMFunctionType;
class PyLLVMModule;

class PyLLVMFunction
{
  public:
    PyLLVMFunction(llvm::Function* obj)
      : obj_(obj) {}

    PyLLVMFunctionType *get_function_type(void);

    static
    PyLLVMFunction *create(PyLLVMFunctionType *function_type, PyLLVMLinkageTypes linkage_type, std::string name, PyLLVMModule *module);
    
    llvm::Function *obj_;
};

} /* namespace pyllvm */

#endif /* PYLLVM_FUNCTION_H */
