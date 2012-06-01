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
    
    PyLLVMValue *create_alloca(PyLLVMType *ty, PyLLVMValue *size, std::string id);

    PyLLVMValue *create_load(PyLLVMValue *ptr);
    
    PyLLVMValue *create_store(PyLLVMValue *val, PyLLVMValue *ptr);
 
    PyLLVMValue *create_gep(PyLLVMValue *ptr, PyLLVMValue *index, std::string id);

    PyLLVMValue *create_add(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateAdd(l->obj_, r->obj_, id));
    }
    
    PyLLVMValue *create_fadd(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateFAdd(l->obj_, r->obj_, id));
    }

    PyLLVMValue *create_sub(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateSub(l->obj_, r->obj_, id));
    }

    PyLLVMValue *create_fsub(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateFSub(l->obj_, r->obj_, id));
    }
 
    PyLLVMValue *create_mul(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateMul(l->obj_, r->obj_, id));
    }
   
    PyLLVMValue *create_fmul(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateFMul(l->obj_, r->obj_, id));
    }
 
    PyLLVMValue *create_sdiv(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateSDiv(l->obj_, r->obj_, id));
    }
   
    PyLLVMValue *create_fdiv(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateFDiv(l->obj_, r->obj_, id));
    }
   
    PyLLVMValue *create_and(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateAnd(l->obj_, r->obj_, id));
    }

    PyLLVMValue *create_or(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateOr(l->obj_, r->obj_, id));
    }

    PyLLVMValue *create_xor(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateXor(l->obj_, r->obj_, id));
    }

    PyLLVMValue *create_icmp_eq(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateICmpEQ(l->obj_, r->obj_, id));
    }
    
    PyLLVMValue *create_icmp_ne(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateICmpNE(l->obj_, r->obj_, id));
    }

    PyLLVMValue *create_icmp_sgt(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateICmpSGT(l->obj_, r->obj_, id));
    }
    
    PyLLVMValue *create_icmp_sge(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateICmpSGE(l->obj_, r->obj_, id));
    }

    PyLLVMValue *create_icmp_slt(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateICmpSLT(l->obj_, r->obj_, id));
    }
    
    PyLLVMValue *create_icmp_sle(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateICmpSLE(l->obj_, r->obj_, id));
    }

    PyLLVMValue *create_fcmp_eq(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateFCmpUEQ(l->obj_, r->obj_, id));
    }
    
    PyLLVMValue *create_fcmp_ne(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateFCmpUNE(l->obj_, r->obj_, id));
    }

    PyLLVMValue *create_fcmp_sgt(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateFCmpUGT(l->obj_, r->obj_, id));
    }
    
    PyLLVMValue *create_fcmp_sge(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateFCmpUGE(l->obj_, r->obj_, id));
    }

    PyLLVMValue *create_fcmp_slt(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateFCmpULT(l->obj_, r->obj_, id));
    }
    
    PyLLVMValue *create_fcmp_sle(PyLLVMValue *l, PyLLVMValue *r, std::string id)
    {
      return new PyLLVMValue(obj_->CreateFCmpULE(l->obj_, r->obj_, id));
    }

    PyLLVMValue *create_call(PyLLVMValue *callee, std::vector<PyLLVMValue *> args, std::string id)
    {
      //llvm::iplist<Value *> args_;
      std::vector<llvm::Value *> args_;
      for (int i=0; i<args.size(); ++i)
      {
        args_.push_back(args[i]->obj_);
      }
      
      return new PyLLVMValue(obj_->CreateCall(callee->obj_, args_, id));
    }
    
    PyLLVMValue *create_br(PyLLVMBasicBlock *d)
    {
      return new PyLLVMValue(obj_->CreateBr(d->obj_));
    }

    PyLLVMValue *create_cond_br(PyLLVMValue *cond, PyLLVMBasicBlock *t, PyLLVMBasicBlock *f)
    {
      return new PyLLVMValue(obj_->CreateCondBr(cond->obj_, t->obj_, f->obj_));
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
