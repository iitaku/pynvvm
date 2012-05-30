#include <llvm/Type.h>

#include "pyllvm-type.h"
#include "pyllvm-context.h"

namespace pyllvm {

PyLLVMType *PyLLVMType::get_int32_ty(PyLLVMContext *context)
{
  return new PyLLVMType(reinterpret_cast<llvm::Type*>(llvm::Type::getInt32Ty(context->obj_)));
}

} /* namespace pyllvm */

