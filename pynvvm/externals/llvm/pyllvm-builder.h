#ifndef PYLLVM_BUILDER_H
#define PYLLVM_BUILDER_H

#include <vector>

#include <llvm/Support/IRBuilder.h>

#include "pyllvm-type.h"
#include "pyllvm-value.h"

namespace pyllvm {

class PyLLVMArgument;
class PyLLVMBasicBlock;
class PyLLVMContext;

class PyLLVMBuilder
{
  public:
    
    static 
    PyLLVMBuilder *create(PyLLVMContext *context);

    PyLLVMBuilder(llvm::IRBuilder<>* obj)
      : obj_(obj) {}

    void set_insert_point(PyLLVMBasicBlock *bb);

    PyLLVMValue *create_call(PyLLVMFunction *function, std::vector<PyLLVMValue* > values, std::string id);

    PyLLVMValue *create_gep(PyLLVMValue *ptr, PyLLVMValue *index, std::string id);
    
    PyLLVMValue *create_load(PyLLVMValue *ptr);
    
    PyLLVMValue *create_store(PyLLVMValue *val, PyLLVMValue *ptr);
     
    PyLLVMValue *create_fadd(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateFAdd(l->obj_, r->obj_, id));
    }

    PyLLVMValue *create_fsub(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateFSub(l->obj_, r->obj_, id));
    }
    
    PyLLVMValue *create_fmul(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateFMul(l->obj_, r->obj_, id));
    }
    
    PyLLVMValue *create_fcmp_ult(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateFCmpULT(l->obj_, r->obj_, id));
    }
 
    PyLLVMValue *create_ui_to_fp(PyLLVMValue *val, PyLLVMType *type, std::string id)
    {
      return new PyLLVMValue(obj_->CreateUIToFP(val->obj_, type->obj_, id));
    }

    PyLLVMValue *create_ret(PyLLVMValue *val)
    {
      return new PyLLVMValue(obj_->CreateRet(val->obj_));
    }

    void create_ret_void(void)
    {
      obj_->CreateRetVoid();
      return;
    }

    llvm::IRBuilder<> *obj_;
};

} /* namespace pyllvm */

#endif /* PYLLVM_BUILDER_H */
