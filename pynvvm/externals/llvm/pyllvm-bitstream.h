#ifndef PYLLVM_BITSTREAM_H
#define PYLLVM_BITSTREAM_H

#include <vector>

#include <llvm/Bitcode/BitstreamWriter.h>
#include <llvm/Bitcode/ReaderWriter.h>

#include "pyllvm-module.h"

namespace pyllvm {

std::string write_bitcode(PyLLVMModule *module)
{
  std::vector<unsigned char> buffer;

  llvm::BitstreamWriter stream(buffer);

  buffer.resize(256*1024);

  llvm::WriteBitcodeToStream(module->obj_, stream);

  std::string code(buffer.begin(), buffer.end());
  
  return code;
}

} /* namespace pyllvm */

#endif /* PYLLVM_BITSTREAM_H */

