#ifndef PYLLVM_BUILDER_H
#define PYLLVM_BUILDER_H

#include <llvm/Support/IRBuilder.h>

namespace pyllvm {

class PyLLVMArgument;
class PyLLVMBasicBlock;
class PyLLVMContext;
class PyLLVMValue;
class PyLLVMValueList;

class PyLLVMBuilder
{
  public:
    
    static 
    PyLLVMBuilder *create(PyLLVMContext *context);

    PyLLVMBuilder(llvm::IRBuilder<>* obj)
      : obj_(obj) {}

    void set_insert_point(PyLLVMBasicBlock *bb);

    PyLLVMValue *create_call(PyLLVMFunction *function, PyLLVMValueList *value_list, std::string id);

    PyLLVMValue *create_gep(PyLLVMValue *value, PyLLVMValueList *value_list, std::string id);
    
    PyLLVMValue *create_load(PyLLVMValue *ptr);
    
    PyLLVMValue *create_store(PyLLVMValue *value, PyLLVMValue *ptr);
    
    void create_ret_void(void);

    llvm::IRBuilder<> *obj_;
};

} /* namespace pyllvm */

#endif /* PYLLVM_BUILDER_H */
