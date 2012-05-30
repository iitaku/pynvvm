#define BOOST_PYTHON_STATIC_LIB

#include <Python.h>

#include <boost/python.hpp>

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
#include "pyllvm-builder.h"
#include "pyllvm-context.h"
#include "pyllvm-module.h"
#include "pyllvm-function.h"
#include "pyllvm-type.h"

#include "pyllvm.h"

namespace py = boost::python;

namespace pyllvm {

BOOST_PYTHON_MODULE(pyllvm)
{
  /* enum */
  py::enum_<PyLLVMLinkageTypes>("linkage_type")
    .value("ExternalLinkage", llvm::GlobalValue::ExternalLinkage)
  ;
    
  /* Argument */
  py::class_<PyLLVMArgument>("argument", py::init<llvm::Argument*>());
  
  /* ArgumentList */
  py::class_<PyLLVMArgument>("argument", py::init<llvm::Argument*>());

  /* Builder */
  py::def("create_builder", &pyllvm::create_builder, py::return_value_policy<py::manage_new_object>());
  py::class_<PyLLVMBuilder>("builder", py::init<llvm::IRBuilder<>*>());

  /* Context */
  py::def("get_global_context", &pyllvm::get_global_context, py::return_value_policy<py::manage_new_object>());
  py::class_<PyLLVMContext>("context", py::init<llvm::LLVMContext&>());
 
  /* Function */
  py::class_<PyLLVMFunction>("function", py::init<llvm::Function*>())
    .def("get_function_type", &PyLLVMFunction::get_function_type, py::return_value_policy<py::manage_new_object>())
  ;
 
  /* FunctionType*/
  py::class_<PyLLVMFunctionType>("function_type", py::init<llvm::FunctionType*>())
    .def("get", &PyLLVMFunctionType::get, py::return_value_policy<py::manage_new_object>())
    .staticmethod("get")
  ;

  /* Module */
  py::def("create_module", &pyllvm::create_module, py::return_value_policy<py::manage_new_object>());
  py::class_<PyLLVMModule>("module", py::init<llvm::Module*>())
    .def("set_data_layout", &PyLLVMModule::set_data_layout)
    .def("dump", &PyLLVMModule::dump)
    .def("get_function", &PyLLVMModule::get_function, py::return_value_policy<py::manage_new_object>())
  ;
  
  /* PointerType*/
  py::class_<PyLLVMPointerType>("pointer_type", py::init<llvm::PointerType*>())
    .def("get", &PyLLVMPointerType::get, py::return_value_policy<py::manage_new_object>())
    .staticmethod("get")
  ;

  /* Type */
  py::class_<PyLLVMType>("type", py::init<llvm::Type*>())
    .def("get_int32_ty", &PyLLVMType::get_int32_ty, py::return_value_policy<py::manage_new_object>())
    .staticmethod("get_int32_ty")
    .def("get_void_ty", &PyLLVMType::get_void_ty, py::return_value_policy<py::manage_new_object>())
    .staticmethod("get_void_ty")
  ;
}

} /* namespace pyllvm */
