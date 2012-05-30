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
    PyLLVMArgumentList(llvm::iplist<llvm::Argument>& obj)
    {
      llvm::iplist<llvm::Argument>::iterator iter;
      for (iter = obj.begin(); iter != obj.end(); ++iter)
      {
        obj_.push_back(new PyLLVMArgument(iter));
      }
    }

    unsigned size(void)
    {
      return static_cast<unsigned>(obj_.size());
    }
    
    PyLLVMArgument *at(unsigned i)
    {
      return obj_.at(i);
    }

    std::vector<PyLLVMArgument*> obj_;
};

} /* namespace pyllvm */

#endif /* PYLLVM_ARGUMENT_H */

