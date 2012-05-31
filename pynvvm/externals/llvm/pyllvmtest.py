import pyllvm

ctx = pyllvm.get_global_context()
bld = pyllvm.builder.create(ctx)
mdl = pyllvm.module.create('test', ctx)

params_ty = [pyllvm.pointer_type.get(pyllvm.type.get_double_ty(ctx), 0), 
             pyllvm.pointer_type.get(pyllvm.type.get_double_ty(ctx), 0)]
ret_ty = pyllvm.type.get_double_ty(ctx)
func_ty = pyllvm.function_type.get(ret_ty, params_ty, False)

fun = pyllvm.function.create(func_ty, pyllvm.linkage_type.external_linkage, 'hoge', mdl)

args = fun.get_arguments()
args[0].set_name('a')
args[1].set_name('b')

bb = pyllvm.basic_block.create(ctx, 'entry', fun)

bld.set_insert_point(bb)

bld.create_gep(args[0], pyllvm.constant_int.get(pyllvm.type.get_int32_ty(ctx), 0), '')

bld.create_ret_void()

nvvmannotate = mdl.get_or_insert_named_metadata('nvvm.annotations')
mdstr = pyllvm.md_string.get(ctx, 'kernel')

md_node = pyllvm.md_node.get(ctx, [fun, mdstr, pyllvm.constant_int.get(pyllvm.type.get_int32_ty(ctx), 1)])

nvvmannotate.add_operand(md_node)

mdl.dump()

