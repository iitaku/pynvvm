#ifndef PYLLVM_TYPE_H
#define PYLLVM_TYPE_H

#include <vector>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/DerivedTypes.h>
#include <llvm/Type.h>

#include "pyllvm-context.h"

namespace pyllvm {

class PyLLVMType
{
  public:
    PyLLVMType(llvm::Type* obj)
      : obj_(obj) {}

    static
    PyLLVMType *get_void_ty(PyLLVMContext *context)
    {
      return new PyLLVMType(reinterpret_cast<llvm::Type*>(llvm::Type::getVoidTy(context->obj_)));
    }
     
    static
    PyLLVMType *get_float_ty(PyLLVMContext *context)
    {
      return new PyLLVMType(reinterpret_cast<llvm::Type*>(llvm::Type::getFloatTy(context->obj_)));
    }

    static
    PyLLVMType *get_double_ty(PyLLVMContext *context)
    {
      return new PyLLVMType(reinterpret_cast<llvm::Type*>(llvm::Type::getDoubleTy(context->obj_)));
    }

    static
    PyLLVMType *get_int1_ty(PyLLVMContext *context)
    {
      return new PyLLVMType(reinterpret_cast<llvm::Type*>(llvm::Type::getInt1Ty(context->obj_)));
    }

    static
    PyLLVMType *get_int8_ty(PyLLVMContext *context)
    {
      return new PyLLVMType(reinterpret_cast<llvm::Type*>(llvm::Type::getInt8Ty(context->obj_)));
    }
    
    static
    PyLLVMType *get_int16_ty(PyLLVMContext *context)
    {
      return new PyLLVMType(reinterpret_cast<llvm::Type*>(llvm::Type::getInt16Ty(context->obj_)));
    }
    
    static
    PyLLVMType *get_int32_ty(PyLLVMContext *context)
    {
      return new PyLLVMType(reinterpret_cast<llvm::Type*>(llvm::Type::getInt32Ty(context->obj_)));
    }

    static
    PyLLVMType *get_int64_ty(PyLLVMContext *context)
    {
      return new PyLLVMType(reinterpret_cast<llvm::Type*>(llvm::Type::getInt64Ty(context->obj_)));
    }

    llvm::Type *obj_;
};

class PyLLVMFunctionType : public PyLLVMType
{
  public:
    PyLLVMFunctionType(llvm::FunctionType* obj)
      : PyLLVMType(obj), obj_(obj) {}

    static
    PyLLVMFunctionType *get(PyLLVMType *result, bool is_var_arg);
    
    static
    PyLLVMFunctionType *get(PyLLVMType *result, std::vector<PyLLVMType *> params, bool is_var_arg);

    llvm::FunctionType *obj_;
};

class PyLLVMPointerType : public PyLLVMType
{
  public:
    PyLLVMPointerType(llvm::PointerType* obj)
      : PyLLVMType(obj), obj_(obj) {}

    static
    PyLLVMPointerType *get(PyLLVMType *type, unsigned address_space);

    llvm::PointerType *obj_;
};

} /* namespace pyllvm */

#endif /* PYLLVMFUNCTION_TYPE_H */
