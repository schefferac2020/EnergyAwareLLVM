import subprocess
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass
class InstructionEnergy:
    """Energy and cycle costs for Cortex-A7 at 1GHz from paper measurements"""
    energy_with_raw: float  # Energy with RAW dependencies (pJ)
    energy_no_raw: float  # Energy without RAW dependencies (pJ)
    latency: float  # Cycles with RAW dependencies
    cpi_no_raw: float  # CPI without RAW dependencies


# Integer arithmetic/logic with register operands (Tables 7.2, 7.4, 7.8)
INTEGER_REG_COSTS = {
    'add': InstructionEnergy(82, 81, 1.0, 1.0),
    'and': InstructionEnergy(69, 70, 1.0, 1.0),
    'eor': InstructionEnergy(72, 71, 1.0, 1.0),
    'mul': InstructionEnergy(146, 78, 3.0, 1.2),
    'orr': InstructionEnergy(72, 71, 1.0, 1.0),
    'rsb': InstructionEnergy(83, 77, 1.0, 1.0),
    'sub': InstructionEnergy(83, 77, 1.0, 1.0),
    'div': InstructionEnergy(221, 152, 5.0, 3.0)
}

# Integer arithmetic/logic with immediate operands (Tables 7.3, 7.6, 7.10)
INTEGER_IMM_COSTS = {
    'add': InstructionEnergy(79, 56, 1.0, 0.5),
    'and': InstructionEnergy(75, 48, 1.0, 0.5),
    'eor': InstructionEnergy(78, 55, 1.0, 0.5),
    'orr': InstructionEnergy(76, 50, 1.0, 0.5),
    'rsb': InstructionEnergy(86, 82, 1.0, 1.0),
    'sub': InstructionEnergy(80, 57, 1.0, 0.5)
}

# Float arithmetic (Tables 7.12, 7.13, 7.14, 7.15)
FLOAT_COSTS = {
    'fadds': InstructionEnergy(199, 93, 4.0, 1.15),
    'fdivs': InstructionEnergy(702, 593, 18.0, 15.0),
    'fmuls': InstructionEnergy(203, 93, 4.0, 1.15),
    'fsubs': InstructionEnergy(200, 93, 4.0, 1.15)
}

# Double arithmetic (Tables 7.17, 7.18, 7.19, 7.20)
DOUBLE_COSTS = {
    'faddd': InstructionEnergy(197, 93, 4.0, 1.28),
    'fdivd': InstructionEnergy(1190, 1083, 32.0, 29.5),
    'fmuld': InstructionEnergy(339, 231, 7.0, 4.0),
    'fsubd': InstructionEnergy(198, 93, 4.0, 1.28)
}

# Integer move instructions (Tables 7.22, 7.24)
INT_MOVE_COSTS = {
    'mov': InstructionEnergy(71, 45, 1.0, 0.5),
    'mvn': InstructionEnergy(84, 78, 1.0, 0.5),
    'mov.imm': InstructionEnergy(49, 49, 1.0, 0.5),
    'mvn.imm': InstructionEnergy(50, 50, 1.0, 0.5)
}

# Float move instructions (Tables 7.27, 7.29)
FLOAT_MOVE_COSTS = {
    'fcpys': InstructionEnergy(198, 93, 4.0, 1.0),
    'fnegs': InstructionEnergy(199, 93, 4.0, 1.0)
}

# Double move instructions (Tables 7.32, 7.34)
DOUBLE_MOVE_COSTS = {
    'fcpyd': InstructionEnergy(206, 86, 4.0, 1.18),
    'fnegd': InstructionEnergy(206, 87, 4.0, 1.18)
}

# Integer compare instructions (Tables 7.36, 7.38)
INT_CMP_COSTS = {
    'cmn': InstructionEnergy(76, 76, 1.0, 1.0),
    'cmp': InstructionEnergy(78, 78, 1.0, 1.0),
    'teq': InstructionEnergy(75, 75, 1.0, 1.0),
    'tst': InstructionEnergy(74, 74, 1.0, 1.0),
    'cmn.imm': InstructionEnergy(52, 52, 1.0, 0.5),
    'cmp.imm': InstructionEnergy(53, 53, 1.0, 0.5),
    'teq.imm': InstructionEnergy(53, 53, 1.0, 0.5),
    'tst.imm': InstructionEnergy(49, 49, 1.0, 0.5)
}

# Float compare instructions (Table 7.40)
FLOAT_CMP_COSTS = {
    'fcmpzs': InstructionEnergy(84, 84, 1.0, 1.0),
    'fcmps': InstructionEnergy(95, 95, 1.0, 1.0)
}

# Double compare instructions (Table 7.42)
DOUBLE_CMP_COSTS = {
    'fcmpzd': InstructionEnergy(89, 89, 1.0, 1.0),
    'fcmpd': InstructionEnergy(103, 103, 1.0, 1.0)
}

# Memory footprint sizes from paper (in KB)
MEM_FOOTPRINTS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Load costs by memory footprint (Table 7.44)
LOAD_COSTS = {
    4: 149,  # L1 cache
    8: 150,
    16: 145,
    32: 142,
    64: 142,
    128: 145,
    256: 157,  # L2 cache
    512: 177,
    1024: 206  # L3 cache
}

# Store costs by memory footprint (Table 7.46)
STORE_COSTS = {
    4: 194,  # L1 cache
    8: 195,
    16: 194,
    32: 192,
    64: 195,
    128: 199,
    256: 210,  # L2 cache
    512: 232,
    1024: 267  # L3 cache
}

# Float/Double memory operations (Tables 7.48, 7.50)
FLOAT_MEM_COSTS = {
    'flds': InstructionEnergy(150, 150, 1.0, 1.0),
    'fldd': InstructionEnergy(196, 196, 1.1, 1.1),
    'fsts': InstructionEnergy(186, 186, 1.75, 1.75),
    'fstd': InstructionEnergy(195, 195, 1.6, 1.6)
}


@dataclass
class EnergyEstimate:
    """Results of energy estimation"""
    total_energy: float
    total_cycles: float
    instruction_counts: Dict[str, int]
    memory_footprint: int  # in KB


def estimate_memory_footprint(asm_code: str) -> int:
    """Estimate memory footprint from assembly code."""
    # Count unique memory addresses referenced
    addresses = set()
    for line in asm_code.splitlines():
        if '[' in line:  # Memory reference
            addr_match = re.search(r'\[.*\]', line)
            if addr_match:
                addresses.add(addr_match.group())

    # Estimate footprint based on unique addresses
    footprint = len(addresses) * 4  # Assume 4 bytes per reference
    # Round up to nearest measured footprint
    for size in MEM_FOOTPRINTS:
        if footprint <= size * 1024:
            return size
    return MEM_FOOTPRINTS[-1]


def classify_instruction(line: str) -> tuple[Optional[str], Optional[InstructionEnergy]]:
    """Classify assembly instruction and return its energy costs."""
    line = line.lower().strip()
    if not line or line.startswith(('.', '#', '@')) or ':' in line:
        return None, None

    # Remove condition codes and flags
    instr = re.sub(r'(eq|ne|cs|cc|mi|pl|vs|vc|hi|ls|ge|lt|gt|le|al|s)$', '', line.split()[0])

    # Check for memory operations
    if instr in ('ldr', 'str'):
        return instr, None  # Handle separately with memory footprint

    # Check float/double memory operations
    if instr in FLOAT_MEM_COSTS:
        return instr, FLOAT_MEM_COSTS[instr]

    # Check immediate vs register operands
    has_immediate = '#' in line

    # Look up in appropriate cost table
    if instr in INTEGER_REG_COSTS and not has_immediate:
        return instr, INTEGER_REG_COSTS[instr]
    elif instr in INTEGER_IMM_COSTS and has_immediate:
        return instr + '.imm', INTEGER_IMM_COSTS[instr]
    elif instr in FLOAT_COSTS:
        return instr, FLOAT_COSTS[instr]
    elif instr in DOUBLE_COSTS:
        return instr, DOUBLE_COSTS[instr]
    elif instr in INT_MOVE_COSTS:
        key = instr + ('.imm' if has_immediate else '')
        return key, INT_MOVE_COSTS[key] if key in INT_MOVE_COSTS else None
    elif instr in FLOAT_MOVE_COSTS:
        return instr, FLOAT_MOVE_COSTS[instr]
    elif instr in DOUBLE_MOVE_COSTS:
        return instr, DOUBLE_MOVE_COSTS[instr]
    elif instr in INT_CMP_COSTS:
        key = instr + ('.imm' if has_immediate else '')
        return key, INT_CMP_COSTS[key] if key in INT_CMP_COSTS else None
    elif instr in FLOAT_CMP_COSTS:
        return instr, FLOAT_CMP_COSTS[instr]
    elif instr in DOUBLE_CMP_COSTS:
        return instr, DOUBLE_CMP_COSTS[instr]

    return None, None


def estimate_program_energy(c_file: Path) -> EnergyEstimate:
    """Estimate energy consumption of a C program."""
    # Compile to ARM assembly
    asm_file = c_file.with_suffix('.s')
    try:
        # Use macOS arm-none-eabi-gcc compiler with Cortex-A7 target
        subprocess.run([
            'arm-none-eabi-gcc',
            '-S',                     # Generate assembly
            '-mcpu=cortex-a7',        # Target Cortex-A7
            '-mthumb',                # Use Thumb instruction set
            '-mfloat-abi=hard',       # Use hardware floating point
            '-march=armv7-a',         # ARMv7-A architecture
            '-O2',                    # Optimization level
            '-o', str(asm_file),      # Output file
            str(c_file)               # Input file
        ], check=True, capture_output=True)
        asm_code = asm_file.read_text()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"Compilation failed: {e}")

    # Estimate memory footprint for load/store costs
    memory_footprint = estimate_memory_footprint(asm_code)

    # Count instructions and calculate energy
    instruction_counts = {}
    total_energy = 0.0
    total_cycles = 0.0

    for line in asm_code.splitlines():
        instr_type, costs = classify_instruction(line)
        if instr_type:
            instruction_counts[instr_type] = instruction_counts.get(instr_type, 0) + 1

            if instr_type == 'ldr':
                energy = LOAD_COSTS[memory_footprint]
                cycles = 1.0
            elif instr_type == 'str':
                energy = STORE_COSTS[memory_footprint]
                cycles = 1.8
            elif costs:
                # Use RAW dependency costs as conservative estimate
                energy = costs.energy_with_raw
                cycles = costs.latency
            else:
                continue

            total_energy += energy
            total_cycles += cycles

    return EnergyEstimate(
        total_energy=total_energy,
        total_cycles=total_cycles,
        instruction_counts=instruction_counts,
        memory_footprint=memory_footprint
    )


def main():
    c_file = Path("test.c")
    try:
        result = estimate_program_energy(c_file)

        print(f"\nEnergy Estimate for {c_file} on Cortex-A7:")
        print(f"Total Energy: {result.total_energy / 1000:.2f} nJ")
        print(f"Total Cycles: {result.total_cycles:.1f}")
        print(f"Estimated Memory Footprint: {result.memory_footprint}KB")
        print("\nInstruction Breakdown:")
        total_instr = sum(result.instruction_counts.values())
        for instr, count in sorted(result.instruction_counts.items()):
            percentage = (count / total_instr) * 100
            print(f"{instr}: {count} ({percentage:.1f}%)")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()