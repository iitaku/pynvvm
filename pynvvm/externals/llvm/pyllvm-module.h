#ifndef PYLLVM_MODULE_H
#define PYLLVM_MODULE_H

#include <string>

#include <llvm/Module.h>

namespace pyllvm {

class PyLLVMContext;
class PyLLVMFunction;

class PyLLVMModule
{
  public:
    PyLLVMModule(llvm::Module* obj)
      : obj_(obj) {}
  
    void set_data_layout(std::string layout);
    
    void dump(void);

    PyLLVMFunction *get_function(std::string fun_name);
  
    llvm::Module* obj_;
};

PyLLVMModule *create_module(std::string id, PyLLVMContext *context);

} /* namespace pyllvm */

#endif /* PYLLVM_MODULE_H */
