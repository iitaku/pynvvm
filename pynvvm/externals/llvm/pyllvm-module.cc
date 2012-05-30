
#include "pyllvm-context.h"
#include "pyllvm-function.h"
#include "pyllvm-metadata.h"
#include "pyllvm-module.h"

namespace pyllvm {

PyLLVMModule *PyLLVMModule::create(std::string id, PyLLVMContext *context)
{
  return new PyLLVMModule(new llvm::Module(id, context->obj_));
}

void PyLLVMModule::set_data_layout(std::string layout)
{
  obj_->setDataLayout(layout);
  return;
}

void PyLLVMModule::dump()
{
  obj_->dump();
  return;
}

PyLLVMFunction *PyLLVMModule::get_function(std::string fun_name)
{
  return new PyLLVMFunction(obj_->getFunction(fun_name));
}
 
PyLLVMNamedMDNode *PyLLVMModule::get_or_insert_named_metadata(std::string name)
{
  return new PyLLVMNamedMDNode(obj_->getOrInsertNamedMetadata(name));
}

} /* namespace pyllvm */

