#ifndef PYLLVM_BITSTREAM_H
#define PYLLVM_BITSTREAM_H

#include <vector>

#include <llvm/Bitcode/BitstreamWriter.h>
#include <llvm/Bitcode/ReaderWriter.h>

namespace pyllvm {

class PyLLVMStringBuffer
{
  public:
    PyLLVMStringBuffer(unsigned n)
      : obj_(n) {}

    std::vector<unsigned char> obj_;
};

class PyLLVMBitstreamWriter
{
  public:
    PyLLVMBitstreamWriter(PyLLVMStringBuffer *buffer)
    {
      //obj_ = new llvm::BitstreamWriter(buffer->obj_);
    }

    llvm::BitstreamWriter *obj_;
};

} /* namespace pyllvm */

#endif /* PYLLVM_BITSTREAM_H */

