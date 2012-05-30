#include <string>
#include <vector>

#include "pyllvm-context.h"
#include "pyllvm-metadata.h"

namespace pyllvm {

PyLLVMMDString *PyLLVMMDString::get(PyLLVMContext *context, std::string id)
{
  return new PyLLVMMDString(llvm::MDString::get(context->obj_, id));
}

PyLLVMMDNode *PyLLVMMDNode::get(PyLLVMContext *context, PyLLVMValueList *value_list)
{
  std::vector<llvm::Value *> value_list_;
  for (unsigned i=0; i<value_list->size(); ++i)
  {
    value_list_.push_back(value_list->at(i)->obj_);
  }
  return new PyLLVMMDNode(llvm::MDNode::get(context->obj_, value_list_));
}


} /* namespace pyllvm */

