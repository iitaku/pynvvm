#ifndef PYLLVM_TYPE_H
#define PYLLVM_TYPE_H

#include <llvm/Type.h>
#include <llvm/DerivedTypes.h>

namespace pyllvm {

class PyLLVMContext;

class PyLLVMType
{
  public:
    PyLLVMType(llvm::Type* obj)
      : obj_(obj) {}

    static
    PyLLVMType *get_int32_ty(PyLLVMContext *context);
    PyLLVMType *get_void_ty(PyLLVMContext *context);

    llvm::Type *obj_;
};

class PyLLVMFunctionType
{
  public:
    PyLLVMFunctionType(llvm::FunctionType* obj)
      : obj_(obj) {}

    static
    PyLLVMFunctionType *get(PyLLVMType *type, bool is_var_arg);

    llvm::FunctionType *obj_;
};

class PyLLVMPointerType
{
  public:
    PyLLVMPointerType(llvm::PointerType* obj)
      : obj_(obj) {}

    static
    PyLLVMPointerType *get(PyLLVMType *type, unsigned address_space);

    llvm::PointerType *obj_;
};

} /* namespace pyllvm */

#endif /* PYLLVMFUNCTION_TYPE_H */
