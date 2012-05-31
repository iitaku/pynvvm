#include <string>
#include <vector>

#include "pyllvm-context.h"
#include "pyllvm-metadata.h"

namespace pyllvm {

PyLLVMMDString *PyLLVMMDString::get(PyLLVMContext *context, std::string id)
{
  return new PyLLVMMDString(llvm::MDString::get(context->obj_, id));
}

PyLLVMMDNode *PyLLVMMDNode::get(PyLLVMContext *context, std::vector<PyLLVMValue *> values)
{
  std::vector<llvm::Value *> values_;
  for (unsigned i=0; i<values.size(); ++i)
  {
    values_.push_back(values.at(i)->obj_);
  }
  
  return new PyLLVMMDNode(llvm::MDNode::get(context->obj_, values_));
}


} /* namespace pyllvm */

