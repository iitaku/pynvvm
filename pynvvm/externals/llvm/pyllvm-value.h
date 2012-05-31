#ifndef PYLLVM_VALUE_H
#define PYLLVM_VALUE_H

#include <vector>

#include <llvm/Value.h>

namespace pyllvm {

class PyLLVMValue
{
  public:
    PyLLVMValue(llvm::Value *obj)
      : obj_(obj) {}
    
    llvm::Value *obj_;
};

//class PyLLVMValueList
//{
//  public:
//    PyLLVMValueList(unsigned n)
//      : obj_(n) {}
//
//    unsigned size(void)
//    {
//      return obj_.size();
//    }
//
//    PyLLVMValue *at(unsigned i)
//    {
//      return obj_.at(i);
//    }
//
//    void push_back(PyLLVMValue *value)
//    {
//      obj_.push_back(value);
//    }
//
//    std::vector<PyLLVMValue*> obj_;
//};

} /* namespace pyllvm */

#endif /* PYLLVM_VALUE_H */

