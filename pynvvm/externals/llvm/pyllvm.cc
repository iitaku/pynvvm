#define BOOST_PYTHON_STATIC_LIB

#include <Python.h>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <llvm/Analysis/Verifier.h>
#include <llvm/Bitcode/BitstreamWriter.h>
#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/DerivedTypes.h>
#include <llvm/Metadata.h>
#include <llvm/Pass.h>
#include <llvm/PassManager.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Support/IRBuilder.h>
#include <llvm/Target/TargetData.h>
#include <llvm/Value.h>

#include <cstdio>
#include <string>
#include <map>
#include <vector>
#include <sstream>

#include "pyllvm-argument.h"
#include "pyllvm-basicblock.h"
#include "pyllvm-bitstream.h"
#include "pyllvm-builder.h"
#include "pyllvm-constants.h"
#include "pyllvm-context.h"
#include "pyllvm-metadata.h"
#include "pyllvm-module.h"
#include "pyllvm-function.h"
#include "pyllvm-type.h"
#include "pyllvm-value.h"

#include "pyllvm.h"

namespace bp = boost::python;

namespace pyllvm {

BOOST_PYTHON_MODULE(pyllvm)
{
  /* enum */
  bp::enum_<PyLLVMLinkageTypes>("linkage_type")
    .value("ExternalLinkage", llvm::GlobalValue::ExternalLinkage)
  ;
    
  /* Argument */
  bp::class_<PyLLVMArgument>("argument", bp::init<llvm::Argument*>());
  
  /* ArgumentList */
  bp::class_<PyLLVMArgumentList>("argument_list", bp::init<llvm::iplist<llvm::Argument>& >())
    .def("size", &PyLLVMArgumentList::size)
    .def("at", &PyLLVMArgumentList::at, bp::return_value_policy<bp::reference_existing_object>())
  ;

  /* BasicBlock */
  bp::class_<PyLLVMBasicBlock>("basic_block", bp::init<llvm::BasicBlock*>())
    .def("create", &PyLLVMBasicBlock::create, bp::return_value_policy<bp::manage_new_object>()).staticmethod("create");

  /* Bitstream */
  bp::class_<PyLLVMStringBuffer>("string_buffer", bp::init<unsigned>());
  bp::class_<PyLLVMBitstreamWriter>("bitstream_writer", bp::init<PyLLVMStringBuffer*>());

  /* Builder */
  bp::class_<PyLLVMBuilder>("builder", bp::init<llvm::IRBuilder<>*>())
    .def("create", &PyLLVMBuilder::create, bp::return_value_policy<bp::manage_new_object>()).staticmethod("create")
    .def("set_insert_point", &PyLLVMBuilder::set_insert_point)
    .def("create_call", &PyLLVMBuilder::create_call, bp::return_value_policy<bp::manage_new_object>())
    .def("create_gep", &PyLLVMBuilder::create_gep, bp::return_value_policy<bp::manage_new_object>())
    .def("create_load", &PyLLVMBuilder::create_load, bp::return_value_policy<bp::manage_new_object>())
    .def("create_store", &PyLLVMBuilder::create_store, bp::return_value_policy<bp::manage_new_object>())
    .def("create_fadd", &PyLLVMBuilder::create_fadd, bp::return_value_policy<bp::manage_new_object>())
    .def("create_fsub", &PyLLVMBuilder::create_fsub, bp::return_value_policy<bp::manage_new_object>())
    .def("create_fmul", &PyLLVMBuilder::create_fmul, bp::return_value_policy<bp::manage_new_object>())
    .def("create_fcmp_ult", &PyLLVMBuilder::create_fcmp_ult, bp::return_value_policy<bp::manage_new_object>())
    .def("create_ui_to_fp", &PyLLVMBuilder::create_ui_to_fp, bp::return_value_policy<bp::manage_new_object>())
    .def("create_create_ret", &PyLLVMBuilder::create_ret, bp::return_value_policy<bp::manage_new_object>())
    .def("create_ret_void", &PyLLVMBuilder::create_ret_void)
  ;

  /* Constants */
  bp::class_<PyLLVMConstantInt, bp::bases<PyLLVMValue> >("constant_int", bp::init<llvm::Constant*>())
    .def("get", &PyLLVMConstantInt::get, bp::return_value_policy<bp::manage_new_object>())
    .staticmethod("get");

  /* Context */
  bp::def("get_global_context", &get_global_context, bp::return_value_policy<bp::manage_new_object>());
  bp::class_<PyLLVMContext>("context", bp::init<llvm::LLVMContext&>());
 
  /* Function */
  bp::class_<PyLLVMFunction, bp::bases<PyLLVMValue> >("function", bp::init<llvm::Function*>())
    .def("get_function_type", &PyLLVMFunction::get_function_type, bp::return_value_policy<bp::manage_new_object>())
    .def("get_argument_list", &PyLLVMFunction::get_argument_list, bp::return_value_policy<bp::manage_new_object>())
  ;
 
  /* FunctionType*/
  bp::class_<PyLLVMFunctionType>("function_type", bp::init<llvm::FunctionType*>())
    .def("get", &PyLLVMFunctionType::get, bp::return_value_policy<bp::manage_new_object>())
    .staticmethod("get")
  ;

  /* Metadata */
  bp::class_<PyLLVMMDString, bp::bases<PyLLVMValue> >("md_string", bp::init<llvm::MDString*>())
    .def("get", &PyLLVMMDString::get, bp::return_value_policy<bp::manage_new_object>())
    .staticmethod("get");

  bp::class_<PyLLVMMDNode, bp::bases<PyLLVMValue> >("md_node", bp::init<llvm::MDNode*>())
    .def("get", &PyLLVMMDNode::get, bp::return_value_policy<bp::manage_new_object>())
    .staticmethod("get");

  bp::class_<PyLLVMNamedMDNode>("named_md_node", bp::init<llvm::NamedMDNode*>())
    .def("add_operand", &PyLLVMNamedMDNode::add_operand)
  ;


  /* Module */
  bp::class_<PyLLVMModule>("module", bp::init<llvm::Module*>())
    .def("create", &PyLLVMModule::create, bp::return_value_policy<bp::manage_new_object>()).staticmethod("create")
    .def("set_data_layout", &PyLLVMModule::set_data_layout)
    .def("dump", &PyLLVMModule::dump)
    .def("get_function", &PyLLVMModule::get_function, bp::return_value_policy<bp::manage_new_object>())
  ;
  
  /* PointerType*/
  bp::class_<PyLLVMPointerType>("pointer_type", bp::init<llvm::PointerType*>())
    .def("get", &PyLLVMPointerType::get, bp::return_value_policy<bp::manage_new_object>())
    .staticmethod("get")
  ;

  /* Type */
  bp::class_<PyLLVMType>("type", bp::init<llvm::Type*>())
    .def("get_int32_ty", &PyLLVMType::get_int32_ty, bp::return_value_policy<bp::manage_new_object>())
    .staticmethod("get_int32_ty")
    .def("get_void_ty", &PyLLVMType::get_void_ty, bp::return_value_policy<bp::manage_new_object>())
    .staticmethod("get_void_ty")
  ;

  /* Value */
  bp::class_<PyLLVMValue>("value", bp::init<llvm::Value*>());
  
  bp::class_<PyLLVMValueList>("value_list", bp::init<unsigned>())
    .def("size", &PyLLVMValueList::size)
    .def("at", &PyLLVMValueList::at, bp::return_value_policy<bp::reference_existing_object>())
    .def("push_back", &PyLLVMValueList::push_back)
  ;
}

} /* namespace pyllvm */
