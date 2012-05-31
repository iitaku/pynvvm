
#include <vector>

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

PyLLVMValue *PyLLVMBuilder::create_call(PyLLVMFunction *function, std::vector<PyLLVMValue *> values, std::string id)
{
  std::vector<llvm::Value*> values_;
  for (unsigned i=0; i<values.size(); ++i)
  {
    values_.push_back(values.at(i)->obj_);
  }

  return new PyLLVMValue(obj_->CreateCall(function->obj_, values_, id));
}

PyLLVMValue *PyLLVMBuilder::create_gep(PyLLVMValue *ptr, PyLLVMValue *index, std::string id)
{
  return new PyLLVMValue(obj_->CreateGEP(ptr->obj_, index->obj_, id));
}

PyLLVMValue *PyLLVMBuilder::create_load(PyLLVMValue *ptr)
{
  return new PyLLVMValue(obj_->CreateLoad(ptr->obj_));
}

PyLLVMValue *PyLLVMBuilder::create_store(PyLLVMValue *value, PyLLVMValue *ptr)
{
  return new PyLLVMValue(obj_->CreateStore(value->obj_, ptr->obj_));
}

} /* namepsace pyllvm */
