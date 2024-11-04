from llvmlite import binding

# Initialize the LLVM binding
binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()

# Define a simple LLVM IR module as a string
ir_code = """
; ModuleID = 'simple_module'
source_filename = "simple.c"
define i32 @main() {
    ret i32 0
}
"""

print(ir_code)
# Parse the LLVM IR text
mod_ir = binding.parse_assembly(ir_code)

# Optional: Verify and print
mod_ir.verify()
print(mod_ir)
