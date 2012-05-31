import pyllvm

ctx = pyllvm.get_global_context()
module = pyllvm.module.create('test', ctx)

params_ty = [pyllvm.type.get_double_ty(ctx), pyllvm.type.get_double_ty(ctx)]
ret_ty = pyllvm.type.get_double_ty(ctx)
func_ty = pyllvm.function_type.get(ret_ty, params_ty, False)

module.dump()

