
#include "pyllvm-basicblock.h"
#include "pyllvm-context.h"
#include "pyllvm-function.h"

namespace pyllvm {

PyLLVMBasicBlock *PyLLVMBasicBlock::create(PyLLVMContext *context, std::string id, PyLLVMFunction *function)
{
  return new PyLLVMBasicBlock(llvm::BasicBlock::Create(context->obj_, id, function->obj_));
}

} /* namespace pyllvm */
