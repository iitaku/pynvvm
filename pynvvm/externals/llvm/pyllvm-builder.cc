
#include "pyllvm-builder.h"
#include "pyllvm-context.h"

namespace pyllvm {

PyLLVMBuilder *create_builder(PyLLVMContext *context)
{
  return new PyLLVMBuilder(new llvm::IRBuilder<>(context->obj_));
}

} /* namepsace pyllvm */
