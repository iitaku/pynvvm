#include "pyllvm-context.h"

namespace pyllvm {

PyLLVMContext *get_global_context()
{
  return new PyLLVMContext(llvm::getGlobalContext());
}

} /* namespace pyllvm */

