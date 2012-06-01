
#include <iostream>
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

PyLLVMValue *PyLLVMBuilder::create_alloca(PyLLVMType *ty, PyLLVMValue *size, std::string id)
{
  return new PyLLVMValue(obj_->CreateAlloca(ty->obj_, size->obj_, id));
}

PyLLVMValue *PyLLVMBuilder::create_load(PyLLVMValue *ptr)
{
  return new PyLLVMValue(obj_->CreateLoad(ptr->obj_));
}

PyLLVMValue *PyLLVMBuilder::create_store(PyLLVMValue *value, PyLLVMValue *ptr)
{
  return new PyLLVMValue(obj_->CreateStore(value->obj_, ptr->obj_));
}

PyLLVMValue *PyLLVMBuilder::create_gep(PyLLVMValue *ptr, PyLLVMValue *index, std::string id)
{
  return new PyLLVMValue(obj_->CreateGEP(ptr->obj_, index->obj_, id));
}

} /* namepsace pyllvm */
