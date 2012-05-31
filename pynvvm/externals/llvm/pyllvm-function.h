#ifndef PYLLVM_FUNCTION_H
#define PYLLVM_FUNCTION_H

#include <string>

#include <llvm/Function.h>

#include "pyllvm.h"
#include "pyllvm-argument.h"
#include "pyllvm-value.h"

namespace pyllvm {

//typedef std::vector<PyLLVMArgument*> PyLLVMArgumentList;
class PyLLVMFunctionType;
class PyLLVMModule;

class PyLLVMFunction : public PyLLVMValue
{
  public:
    static
    PyLLVMFunction *create(PyLLVMFunctionType *function_type, PyLLVMLinkageTypes linkage_type, std::string name, PyLLVMModule *module);

    PyLLVMFunction(llvm::Function* obj);

    PyLLVMFunctionType *get_function_type(void);
 
    std::vector<PyLLVMArgument*> get_arguments(void);

    std::string get_name(void)
    {
      return obj_->getName();
    }
  
    void erase_from_parent(void)
    {
      obj_->eraseFromParent();
      return;
    }
        
    llvm::Function *obj_;

    //PyLLVMArgumentList argument_list_;
};

} /* namespace pyllvm */

#endif /* PYLLVM_FUNCTION_H */
