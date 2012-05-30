
#include "pyllvm-argument.h"
#include "pyllvm-basicblock.h"
#include "pyllvm-builder.h"
#include "pyllvm-context.h"
#include "pyllvm-function.h"
#include "pyllvm-value.h"

namespace pyllvm {

PyLLVMBuilder *PyLLVMBuilder::create(PyLLVMContext *context)
{
  return new PyLLVMBuilder(new llvm::IRBuilder<>(context->obj_));
}

void PyLLVMBuilder::set_insert_point(PyLLVMBasicBlock *bb)
{
  obj_->SetInsertPoint(bb->obj_);
  return;
}

PyLLVMValue *PyLLVMBuilder::create_call(PyLLVMFunction *function, PyLLVMValueList *value_list, std::string id)
{
  std::vector<llvm::Value*> value_list_;
  for (unsigned i=0; i<value_list->obj_.size(); ++i)
  {
    value_list_.push_back(value_list->obj_.at(i)->obj_);
  }

  return new PyLLVMValue(obj_->CreateCall(function->obj_, value_list_, id));
}

PyLLVMValue *PyLLVMBuilder::create_gep(PyLLVMValue *value, PyLLVMValueList *value_list, std::string id)
{
  std::vector<llvm::Value*> value_list_;
  for (unsigned i=0; i<value_list->obj_.size(); ++i)
  {
    value_list_.push_back(value_list->obj_.at(i)->obj_);
  }

  return new PyLLVMValue(obj_->CreateGEP(value->obj_, value_list_, id));
}

PyLLVMValue *PyLLVMBuilder::create_load(PyLLVMValue *ptr)
{
  return new PyLLVMValue(obj_->CreateLoad(ptr->obj_));
}

PyLLVMValue *PyLLVMBuilder::create_store(PyLLVMValue *value, PyLLVMValue *ptr)
{
  return new PyLLVMValue(obj_->CreateStore(value->obj_, ptr->obj_));
}

void PyLLVMBuilder::create_ret_void(void)
{
  obj_->CreateRetVoid();
  return;
}

} /* namepsace pyllvm */
