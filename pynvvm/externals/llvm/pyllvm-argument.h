#ifndef PYLLVM_ARGUMENT_H
#define PYLLVM_ARGUMENT_H

#include <vector>

#include <llvm/Argument.h>
#include <llvm/Function.h>

namespace pyllvm {

class PyLLVMArgument
{
  public:
    PyLLVMArgument(llvm::Argument* obj)
      : obj_(obj) {}

    llvm::Argument *obj_;
};

class PyLLVMArgumentList
{
  public:
    PyLLVMArgumentList(llvm::iplist<llvm::Argument> *obj_)
    {
      llvm::iplist<llvm::Argument>::iterator iter;
      for (iter = obj_->begin(); iter != obj_->end(); ++iter)
      {
        obj.push_back(new PyLLVMArgument(iter));
      }
    }

    std::vector<PyLLVMArgument*> obj;
};

} /* namespace pyllvm */

#endif /* PYLLVM_ARGUMENT_H */

