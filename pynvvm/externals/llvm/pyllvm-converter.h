#ifndef PYLLVM_CONVERTER_H
#define PYLLVM_CONVERTER_H

#include <boost/python.hpp>
#include <boost/foreach.hpp>
#include <boost/range/value_type.hpp>

namespace pyllvm {

template<typename T_>
class vector_to_pylist_converter {
public:
    typedef T_ native_type;

    static PyObject* convert(native_type const& v)
    {
      namespace bp = boost::python;
      bp::list retval;
      BOOST_FOREACH(typename boost::range_value<native_type>::type i, v)
      {
        retval.append(bp::object(i));
      }
      return bp::incref(retval.ptr());
    }
};

template<typename T_>
class pylist_to_vector_converter 
{
  public:
    typedef T_ native_type;

    static void* convertible(PyObject* pyo)
    {
      if (!PySequence_Check(pyo))
        return 0;

      return pyo;
    }

    static void construct(PyObject* pyo, boost::python::converter::rvalue_from_python_stage1_data* data)
    {
      namespace bp = boost::python;
      native_type* storage = new(reinterpret_cast<bp::converter::rvalue_from_python_storage<native_type>*>(data)->storage.bytes) native_type();
      for (bp::ssize_t i = 0, l = PySequence_Size(pyo); i < l; ++i)
      {
        storage->push_back(
            bp::extract<typename boost::range_value<native_type>::type>(
              PySequence_GetItem(pyo, i)));
      }
      data->convertible = storage;
    }
};

} /* namespace pyllvm */

#endif /* PYLLVM_CONVERTER_H */
