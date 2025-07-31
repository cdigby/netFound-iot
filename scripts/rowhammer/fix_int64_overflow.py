import pyarrow as pa
import numpy as np

# --- Configuration ---
# The input file that was corrupted by the bit-flipping attack.
INPUT_FILE = "../data/iot2023-8class-http-5m.arrow"
# The output file where the corrected data will be saved.
OUTPUT_FILE = "../data/iot2023-8class-http-5m-fixed.arrow"

# --- Constants ---
# The maximum value for a signed 64-bit integer (int64).
# Any uint64 value larger than this will cause an overflow when cast to int64.
# This corresponds to 2**63 - 1.
INT64_MAX = np.iinfo(np.int64).max

# The value used to flip the most significant bit (MSB) of a uint64 number.
# In binary, this is a 1 followed by 63 zeros (1000...000).
# A bitwise XOR with this value flips the 64th bit.
MSB_FLIP_MASK = 1 << 63

print(f"Loading data from: {INPUT_FILE}")

try:
    # Load the entire table from the input Arrow file into memory.
    with pa.OSFile(INPUT_FILE, "rb") as f:
        with pa.ipc.open_stream(f) as reader:
            table = reader.read_all()

    print("Data loaded successfully. Analyzing and fixing columns...")

    # Create a dictionary to hold the columns for the new, corrected table.
    # We start by populating it with all columns from the original table.
    new_columns = {name: table[name] for name in table.column_names}
    
    corrected_column_count = 0

    # Iterate over each column in the table to find and fix uint64 overflows.
    for i, field in enumerate(table.schema):
        column_name = field.name
        column_type = field.type

        # Check if the column is of type uint64 or a list of uint64.
        # This is where the overflow is expected to occur.
        is_uint64_type = pa.types.is_uint64(column_type)
        is_list_of_uint64 = pa.types.is_list(column_type) and pa.types.is_uint64(column_type.value_type)

        if is_uint64_type:
            print(f"-> Checking uint64 column: '{column_name}'")
            # Convert the PyArrow column to a NumPy array, creating a writable copy.
            # The .copy() is crucial to avoid a "read-only" error.
            numpy_array = table[i].to_numpy().copy()

            # Create a boolean mask to identify values that are larger than the int64 max.
            overflow_mask = numpy_array > INT64_MAX
            
            num_overflows = np.sum(overflow_mask)
            if num_overflows > 0:
                print(f"   Found {num_overflows} values exceeding int64 limit. Fixing them...")
                # Apply the bitwise XOR operation only on the elements identified by the mask.
                # This flips the MSB, bringing the value back into the valid int64 range.
                numpy_array[overflow_mask] ^= MSB_FLIP_MASK
                
                # Replace the old column in our dictionary with the new, corrected PyArrow array.
                new_columns[column_name] = pa.array(numpy_array, type=pa.uint64())
                corrected_column_count += 1
            else:
                print("   No overflow values found in this column.")

        elif is_list_of_uint64:
            print(f"-> Checking list<uint64> column: '{column_name}'")
            # For list types, we must iterate through each list (row) individually.
            corrected_lists = []
            column_data = table[i].to_pylist() # Convert to list of lists/arrays
            
            overflows_in_col = 0
            for sublist in column_data:
                if sublist is None:
                    corrected_lists.append(None)
                    continue
                
                # Convert the inner list to a NumPy array to perform the check.
                # np.array() creates a writable copy by default.
                numpy_array = np.array(sublist, dtype=np.uint64)
                overflow_mask = numpy_array > INT64_MAX
                
                num_overflows = np.sum(overflow_mask)
                if num_overflows > 0:
                    overflows_in_col += num_overflows
                    # Fix the values in place using the mask.
                    numpy_array[overflow_mask] ^= MSB_FLIP_MASK
                
                corrected_lists.append(numpy_array.tolist())

            if overflows_in_col > 0:
                 print(f"   Found and fixed {overflows_in_col} total overflow values across all lists.")
                 # Re-create the PyArrow array from the corrected Python lists.
                 new_columns[column_name] = pa.array(corrected_lists, type=column_type)
                 corrected_column_count += 1
            else:
                print("   No overflow values found in this column.")


    if corrected_column_count > 0:
        # Create a new table from the corrected columns, preserving the original schema.
        new_table = pa.Table.from_pydict(new_columns, schema=table.schema)

        # Write the fixed table to the new output file.
        print(f"\nWriting corrected data to: {OUTPUT_FILE}")
        with pa.OSFile(OUTPUT_FILE, "wb") as f:
            with pa.ipc.new_stream(f, new_table.schema) as writer:
                writer.write_table(new_table)
        print("Correction complete. New Arrow file has been saved.")
    else:
        print("\nNo columns required correction. Output file not created.")

except FileNotFoundError:
    print(f"Error: The input file '{INPUT_FILE}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

