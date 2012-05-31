#ifndef PYLLVM_ARGUMENT_H
#define PYLLVM_ARGUMENT_H

#include <vector>

#include <llvm/Argument.h>
#include <llvm/Function.h>

#include "pyllvm-value.h"

namespace pyllvm {

class PyLLVMArgument : public PyLLVMValue
{
  public:
    PyLLVMArgument(llvm::Argument* obj)
      : PyLLVMValue(obj), obj_(obj) {}

    void set_name(std::string id)
    {
      obj_->setName(id);
      return;
    }
    
    llvm::Argument *obj_;
};

} /* namespace pyllvm */

#endif /* PYLLVM_ARGUMENT_H */

