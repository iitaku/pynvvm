#ifndef PYLLVM_METADATA_H
#define PYLLVM_METADATA_H

#include <string>
#include <vector>

#include <llvm/Metadata.h>

#include "pyllvm-value.h"

namespace pyllvm {

class PyLLVMContext;

class PyLLVMMDString : public PyLLVMValue
{
  public:
    PyLLVMMDString(llvm::MDString *obj)
      : PyLLVMValue(obj), obj_(obj) {}

    static
    PyLLVMMDString *get(PyLLVMContext *context, std::string id);

    llvm::MDString *obj_;
};

class PyLLVMMDNode : public PyLLVMValue
{
  public:
    PyLLVMMDNode(llvm::MDNode *obj)
      : PyLLVMValue(obj), obj_(obj) {}

    static
    //PyLLVMMDNode *get(PyLLVMContext *context, PyLLVMValueList *value_list);
    PyLLVMMDNode *get(PyLLVMContext *context, std::vector<PyLLVMValue *> values);

    llvm::MDNode *obj_;
};

class PyLLVMNamedMDNode
{
  public:
    PyLLVMNamedMDNode(llvm::NamedMDNode *obj)
      : obj_(obj) {}
   
    void add_operand(PyLLVMMDNode *node)
    {
      obj_->addOperand(node->obj_);
      return;
    }
    
    llvm::NamedMDNode* obj_;
};

} /* namespace pyllvm */

#endif /* PYLLVM_METADATA_H */
