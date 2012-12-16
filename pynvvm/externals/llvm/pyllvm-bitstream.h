#ifndef PYLLVM_BITSTREAM_H
#define PYLLVM_BITSTREAM_H

#include <iostream>
#include <vector>

#include <llvm/Analysis/Verifier.h>
#include <llvm/Bitcode/BitstreamWriter.h>
#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/Pass.h>
#include <llvm/PassManager.h>
#include <llvm/Support/raw_ostream.h>

#include "pyllvm-module.h"

namespace pyllvm {

std::string print_ir(PyLLVMModule *module)
{
#if 0
  llvm::PassManager *PM;
  int error = 0;
  PM = new llvm::PassManager();
  PM->add(llvm::createVerifierPass(llvm::ReturnStatusAction));
  if (PM->run(*(module->obj_))) {
    error = 1;
  }
   
  delete PM;
  
  std::vector<unsigned char> buffer;

  llvm::BitstreamWriter stream(buffer);

  buffer.resize(256*1024);

  llvm::WriteBitcodeToStream(module->obj_, stream);

  //std::string code(buffer.begin(), buffer.end());
#endif
  std::string code;
 
  llvm::raw_string_ostream ss(code);
  module->obj_->print(ss, NULL);
  
  return code;
}

} /* namespace pyllvm */

#endif /* PYLLVM_BITSTREAM_H */

