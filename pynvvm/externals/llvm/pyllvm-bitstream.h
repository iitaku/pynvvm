#ifndef PYLLVM_BITSTREAM_H
#define PYLLVM_BITSTREAM_H

#include <iostream>
#include <vector>

#include <llvm/Analysis/Verifier.h>
#include <llvm/Bitcode/BitstreamWriter.h>
#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/Pass.h>
#include <llvm/PassManager.h>

#include "pyllvm-module.h"

namespace pyllvm {

std::string write_bitcode(PyLLVMModule *module)
{
  llvm::PassManager *PM;
  int error = 0;
  PM = new llvm::PassManager();
  PM->add(llvm::createVerifierPass(llvm::ReturnStatusAction));
  if (PM->run(*(module->obj_))) {
    error = 1;
  }
  std::cout << error << std::endl;
  
  module->obj_->dump();
  
  delete PM;
  
  std::vector<unsigned char> buffer;

  llvm::BitstreamWriter stream(buffer);

  buffer.resize(256*1024);

  llvm::WriteBitcodeToStream(module->obj_, stream);

  std::string code(buffer.begin(), buffer.end());
 
  return code;
}

} /* namespace pyllvm */

#endif /* PYLLVM_BITSTREAM_H */

