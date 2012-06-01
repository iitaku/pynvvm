#define BOOST_PYTHON_STATIC_LIB

#include <Python.h>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <string>
#include <vector>

#include "pyllvm-argument.h"
#include "pyllvm-basicblock.h"
#include "pyllvm-bitstream.h"
#include "pyllvm-builder.h"
#include "pyllvm-constants.h"
#include "pyllvm-converter.h"
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

  /* Type */
  bp::class_<PyLLVMType>("type", bp::init<llvm::Type*>())
    .def("get_void_ty", &PyLLVMType::get_void_ty, bp::return_value_policy<bp::manage_new_object>())
    .staticmethod("get_void_ty")
    .def("get_float_ty", &PyLLVMType::get_float_ty, bp::return_value_policy<bp::manage_new_object>())
    .staticmethod("get_float_ty")
    .def("get_double_ty", &PyLLVMType::get_double_ty, bp::return_value_policy<bp::manage_new_object>())
    .staticmethod("get_double_ty")
    .def("get_int1_ty", &PyLLVMType::get_int1_ty, bp::return_value_policy<bp::manage_new_object>())
    .staticmethod("get_int1_ty")
    .def("get_int8_ty", &PyLLVMType::get_int8_ty, bp::return_value_policy<bp::manage_new_object>())
    .staticmethod("get_int8_ty")
    .def("get_int16_ty", &PyLLVMType::get_int16_ty, bp::return_value_policy<bp::manage_new_object>())
    .staticmethod("get_int16_ty")
    .def("get_int32_ty", &PyLLVMType::get_int32_ty, bp::return_value_policy<bp::manage_new_object>())
    .staticmethod("get_int32_ty")
    .def("get_int64_ty", &PyLLVMType::get_int64_ty, bp::return_value_policy<bp::manage_new_object>())
    .staticmethod("get_int64_ty")
  ;

  /* TypeVector */
  typedef std::vector<PyLLVMType *> PyLLVMTypeVector;
  
  bp::to_python_converter<PyLLVMTypeVector, vector_to_pylist_converter<PyLLVMTypeVector> >();
  bp::converter::registry::push_back(
    &pylist_to_vector_converter<PyLLVMTypeVector>::convertible,
    &pylist_to_vector_converter<PyLLVMTypeVector>::construct,
    bp::type_id<PyLLVMTypeVector>())
  ;
   
  /* PointerType*/
  bp::class_<PyLLVMPointerType, bp::bases<PyLLVMType> >("pointer_type", bp::init<llvm::PointerType*>())
    .def("get", &PyLLVMPointerType::get, bp::return_value_policy<bp::manage_new_object>())
    .staticmethod("get")
  ;

  /* FunctionType*/
  bp::class_<PyLLVMFunctionType, bp::bases<PyLLVMType> >("function_type", bp::init<llvm::FunctionType*>())
    .def("get", (PyLLVMFunctionType *(*)(PyLLVMType*, bool))&PyLLVMFunctionType::get, bp::return_value_policy<bp::manage_new_object>())
    .def("get", (PyLLVMFunctionType *(*)(PyLLVMType*, std::vector<PyLLVMType *>, bool))&PyLLVMFunctionType::get, bp::return_value_policy<bp::manage_new_object>())
    .staticmethod("get")
  ;

  /* Value */
  bp::class_<PyLLVMValue>("value", bp::init<llvm::Value*>());

  /* ValueVector */
  typedef std::vector<PyLLVMValue *> PyLLVMValueVector;
  
  bp::to_python_converter<PyLLVMValueVector, vector_to_pylist_converter<PyLLVMValueVector> >();
  bp::converter::registry::push_back(
    &pylist_to_vector_converter<PyLLVMValueVector>::convertible,
    &pylist_to_vector_converter<PyLLVMValueVector>::construct,
    bp::type_id<PyLLVMValueVector>())
  ;

  /* Argument */
  bp::class_<PyLLVMArgument, bp::bases<PyLLVMValue> >("argument", bp::init<llvm::Argument*>())
    .def("set_name", &PyLLVMArgument::set_name)
  ;
  
  /* ArgumentVector */
  typedef std::vector<PyLLVMArgument *> PyLLVMArgumentVector;
  bp::to_python_converter<PyLLVMArgumentVector, vector_to_pylist_converter<PyLLVMArgumentVector> >();
  bp::converter::registry::push_back(
    &pylist_to_vector_converter<PyLLVMArgumentVector>::convertible,
    &pylist_to_vector_converter<PyLLVMArgumentVector>::construct,
    bp::type_id<PyLLVMArgumentVector>())
  ;

  /* Constants */
  bp::class_<PyLLVMConstantInt, bp::bases<PyLLVMValue> >("constant_int", bp::init<llvm::Constant*>())
    .def("get", &PyLLVMConstantInt::get, bp::return_value_policy<bp::manage_new_object>())
    .staticmethod("get")
    ;

  bp::class_<PyLLVMConstantFP, bp::bases<PyLLVMValue> >("constant_fp", bp::init<llvm::Constant*>())
    .def("get", &PyLLVMConstantFP::get, bp::return_value_policy<bp::manage_new_object>())
    .staticmethod("get")
    ;

  /* Function */
  bp::class_<PyLLVMFunction, bp::bases<PyLLVMValue> >("function", bp::init<llvm::Function*>())
    .def("create", &PyLLVMFunction::create, bp::return_value_policy<bp::manage_new_object>()).staticmethod("create")
    .def("get_function_type", &PyLLVMFunction::get_function_type, bp::return_value_policy<bp::manage_new_object>())
    .def("get_arguments", &PyLLVMFunction::get_arguments)
    .def("get_name", &PyLLVMFunction::get_name)
    .def("erase_from_parent", &PyLLVMFunction::erase_from_parent)
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
    .def("get_or_insert_named_metadata", &PyLLVMModule::get_or_insert_named_metadata, bp::return_value_policy<bp::manage_new_object>())
  ;

  /* BasicBlock */
  bp::class_<PyLLVMBasicBlock>("basic_block", bp::init<llvm::BasicBlock*>())
    .def("create", &PyLLVMBasicBlock::create, bp::return_value_policy<bp::manage_new_object>()).staticmethod("create");
  
  /* Builder */
  bp::class_<PyLLVMBuilder>("builder", bp::init<llvm::IRBuilder<>*>())
    .def("create", &PyLLVMBuilder::create, bp::return_value_policy<bp::manage_new_object>()).staticmethod("create")
    .def("set_insert_point", &PyLLVMBuilder::set_insert_point)
    .def("create_alloca", &PyLLVMBuilder::create_alloca, bp::return_value_policy<bp::manage_new_object>())
    .def("create_load", &PyLLVMBuilder::create_load, bp::return_value_policy<bp::manage_new_object>())
    .def("create_store", &PyLLVMBuilder::create_store, bp::return_value_policy<bp::manage_new_object>())
    .def("create_gep", &PyLLVMBuilder::create_gep, bp::return_value_policy<bp::manage_new_object>())
    .def("create_add", &PyLLVMBuilder::create_add, bp::return_value_policy<bp::manage_new_object>())
    .def("create_fadd", &PyLLVMBuilder::create_fadd, bp::return_value_policy<bp::manage_new_object>())
    .def("create_sub", &PyLLVMBuilder::create_sub, bp::return_value_policy<bp::manage_new_object>())
    .def("create_fsub", &PyLLVMBuilder::create_fsub, bp::return_value_policy<bp::manage_new_object>())
    .def("create_mul", &PyLLVMBuilder::create_mul, bp::return_value_policy<bp::manage_new_object>())
    .def("create_fmul", &PyLLVMBuilder::create_fmul, bp::return_value_policy<bp::manage_new_object>())
    .def("create_sdiv", &PyLLVMBuilder::create_sdiv, bp::return_value_policy<bp::manage_new_object>())
    .def("create_fdiv", &PyLLVMBuilder::create_fdiv, bp::return_value_policy<bp::manage_new_object>())
    .def("create_and", &PyLLVMBuilder::create_and, bp::return_value_policy<bp::manage_new_object>())
    .def("create_or", &PyLLVMBuilder::create_or, bp::return_value_policy<bp::manage_new_object>())
    .def("create_xor", &PyLLVMBuilder::create_xor, bp::return_value_policy<bp::manage_new_object>())
    .def("create_icmp_eq", &PyLLVMBuilder::create_icmp_eq, bp::return_value_policy<bp::manage_new_object>())
    .def("create_icmp_ne", &PyLLVMBuilder::create_icmp_ne, bp::return_value_policy<bp::manage_new_object>())
    .def("create_icmp_sgt", &PyLLVMBuilder::create_icmp_sgt, bp::return_value_policy<bp::manage_new_object>())
    .def("create_icmp_sge", &PyLLVMBuilder::create_icmp_sge, bp::return_value_policy<bp::manage_new_object>())
    .def("create_icmp_slt", &PyLLVMBuilder::create_icmp_slt, bp::return_value_policy<bp::manage_new_object>())
    .def("create_icmp_sle", &PyLLVMBuilder::create_icmp_sle, bp::return_value_policy<bp::manage_new_object>())
    .def("create_fcmp_eq", &PyLLVMBuilder::create_fcmp_eq, bp::return_value_policy<bp::manage_new_object>())
    .def("create_fcmp_ne", &PyLLVMBuilder::create_fcmp_ne, bp::return_value_policy<bp::manage_new_object>())
    .def("create_fcmp_sgt", &PyLLVMBuilder::create_fcmp_sgt, bp::return_value_policy<bp::manage_new_object>())
    .def("create_fcmp_sge", &PyLLVMBuilder::create_fcmp_sge, bp::return_value_policy<bp::manage_new_object>())
    .def("create_fcmp_slt", &PyLLVMBuilder::create_fcmp_slt, bp::return_value_policy<bp::manage_new_object>())
    .def("create_fcmp_sle", &PyLLVMBuilder::create_fcmp_sle, bp::return_value_policy<bp::manage_new_object>())
    .def("create_call", &PyLLVMBuilder::create_call, bp::return_value_policy<bp::manage_new_object>())
    .def("create_br", &PyLLVMBuilder::create_br, bp::return_value_policy<bp::manage_new_object>())
    .def("create_cond_br", &PyLLVMBuilder::create_cond_br, bp::return_value_policy<bp::manage_new_object>())
    .def("create_create_ret", &PyLLVMBuilder::create_ret, bp::return_value_policy<bp::manage_new_object>())
    .def("create_ret_void", &PyLLVMBuilder::create_ret_void)
  ;

  /* Context */
  bp::def("get_global_context", &get_global_context, bp::return_value_policy<bp::manage_new_object>());
  bp::class_<PyLLVMContext>("context", bp::init<llvm::LLVMContext&>());

  /* Bitstream */
  bp::def("write_bitcode", &write_bitcode);

  /* enum */
  bp::enum_<PyLLVMLinkageTypes>("linkage_type")
    .value("external_linkage", llvm::GlobalValue::ExternalLinkage)
  ;
}

} /* namespace pyllvm */
