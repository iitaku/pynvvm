#ifndef PYLLVM_MODULE_H
#define PYLLVM_MODULE_H

#include <string>

#include <llvm/Module.h>

namespace pyllvm {

class PyLLVMContext;
class PyLLVMFunction;
class PyLLVMNamedMDNode;

class PyLLVMModule
{
  public:
    static
    PyLLVMModule *create(std::string id, PyLLVMContext *context);

    PyLLVMModule(llvm::Module* obj)
      : obj_(obj) {}
  
    void set_data_layout(std::string layout);
    
    void dump(void);

    PyLLVMFunction *get_function(std::string fun_name);
 
    PyLLVMNamedMDNode *get_or_insert_named_metadata(std::string name);
  
    llvm::Module* obj_;
};

} /* namespace pyllvm */

#endif /* PYLLVM_MODULE_H */
