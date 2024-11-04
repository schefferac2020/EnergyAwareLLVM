import llvmlite.ir as ir
import llvmlite.binding as binding

with open("test1.ll",'r') as f:
    the_ir = str(f.read())
    
mod_bc=binding.parse_assembly(the_ir)

# Iterate over functions in the module
for function in mod_bc.functions:
    print("Function:", function.name)

    # Iterate over basic blocks in the function
    for basic_block in function.blocks:
        print("Basic Block:", basic_block.name)

        # Iterate over instructions in the basic block (if needed)
        for instruction in basic_block.instructions:
            print("Instruction:", instruction)
